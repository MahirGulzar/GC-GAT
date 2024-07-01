from models.encoders.encoder import PredictionEncoder
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from typing import Dict, Tuple
import math


# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import numpy as np


class PolylineSubgraph(nn.Module):

    def __init__(self, args: Dict):
        """
        Polyline subgraph encoder from VectorNet (Gao et al., CVPR 2020).
        Has N encoder layers. Each layer encodes every feature in a polyline using an MLP with shared
        weights, followed by a permutation invariant aggregation operator (element-wise max used in the paper).
        Aggregated vector is concatenated with each independent feature encoding.
        Layer is repeated N times. Final encodings are passed through the permutation invariant
        aggregation operator to give polyline encodings.

        args to include
            'num_layers': int Number of repeated encoder layers
            'mlp_size':  int Width of MLP hidden layer
            'lane_feat_size': int Lane feature dimension
            'agent_feat_size': int Agent feature dimension

        """
        super().__init__()
        self.num_layers = args['num_layers']
        self.mlp_size = args['mlp_size']
        self.feat_size = args['feat_size']

        # Encoder layers

        """
        Note: I'm not completely sure if VectorNet uses different MLPs for agents, map polylines and map polygons.
        The paper doesn't seem to mention this clearly. However, agents and map polylines will typically have different 
        attribute features. At least the first linear layer has to be different. 
        Shouldn't affect the global attention aggregator. All final feats will have the same dimensions.
        """

        polyline_encoders = [nn.Linear(self.feat_size + 2, self.mlp_size)]
        for n in range(1, self.num_layers):
            polyline_encoders.append(nn.Linear(self.mlp_size*2, self.mlp_size))
        self.polyline_encoders = nn.ModuleList(polyline_encoders)

        # Layer norm and relu
        self.layer_norm = nn.LayerNorm(self.mlp_size)
        self.relu = nn.ReLU()

    def forward(self, features: torch.Tensor, masks: torch.Tensor) -> Dict:

        # Encode polyline features
        features = self.convert2vectornet_feat_format(features)
        masks = masks[:, :, :-1, :]
        features_enc, masks = self.encode(self.polyline_encoders, features, masks)

        return {'features_enc':features_enc, 'masks': masks}

    def encode(self, encoder_layers: nn.ModuleList, input_feats: torch.Tensor,
               masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies encoding layers to a given set of input feats
        """
        masks = masks[..., 0]
        masks[masks == 1] = -math.inf

        encodings = input_feats
        for n in range(len(encoder_layers)):
            encodings = self.relu(self.layer_norm(encoder_layers[n](encodings)))
            encodings = encodings + masks.unsqueeze(-1)
            agg_enc, _ = torch.max(encodings, dim=2)
            encodings = torch.cat((encodings, agg_enc.unsqueeze(2).repeat(1, 1, encodings.shape[2], 1)), dim=3)
            encodings[encodings == -math.inf] = 0

        agg_encoding, _ = torch.max(encodings, dim=2)
        masks[masks == -math.inf] = 1

        return agg_encoding, masks[..., 0]

    @staticmethod
    def convert2vectornet_feat_format(feats: torch.Tensor) -> torch.Tensor:
        """
        Helper function to convert a tensor of node features to the vectornet format.
        By default the datasets return node features of the format [x, y, attribute feats...].
        Vectornet uses the following format [x, y, x_next, y_next, attribute_feats]
        :param feats: Tensor of feats, shape [batch_size, max_polylines, max_len, feat_dim]
        :return: Tensor of updated feats, shape [batch_size, max_polylines, max_len, feat_dim + 2]
        """
        xy = feats[:, :, :-1, :2]
        xy_next = feats[:, :, 1:, :2]
        attr = feats[:, :, :-1, 2:]
        feats = torch.cat((xy, xy_next, attr), dim=3)
        return feats


class PGPModEncoder(PredictionEncoder):

    def __init__(self, args: Dict):
        super().__init__()

        ############################ <Agents> ################################

        # Target agent encoder
        self.target_agent_emb = nn.Linear(args['target_agent_feat_size'], args['target_agent_emb_size'])
        self.target_agent_enc = nn.GRU(args['target_agent_emb_size'], args['target_agent_enc_size'], batch_first=True)

        # Surrounding agent encoder
        self.nbr_emb = nn.Linear(args['nbr_feat_size'] + 1, args['nbr_emb_size'])
        self.nbr_enc = nn.GRU(args['nbr_emb_size'], args['nbr_enc_size'], batch_first=True)


        ############################ <Map Elements> ################################

        # Node encoders
        self.node_emb = nn.Linear(args['node_feat_size'], args['node_emb_size'])
        self.node_encoder = nn.GRU(args['node_emb_size'], args['node_enc_size'], batch_first=True)

        # Agent-node attention
        self.query_emb = nn.Linear(args['node_enc_size'], args['node_enc_size'])
        self.key_emb = nn.Linear(args['nbr_enc_size'], args['node_enc_size'])
        self.val_emb = nn.Linear(args['nbr_enc_size'], args['node_enc_size'])
        self.a_n_att = nn.MultiheadAttention(args['node_enc_size'], num_heads=1)
        self.a_n_mix = nn.Linear(args['node_enc_size']*2, args['node_enc_size'])


        # Intersection-node attention
        self.i_n_query_emb = nn.Linear(args['node_enc_size'], args['node_enc_size'])
        self.i_n_key_emb = nn.Linear(args['node_enc_size'], args['node_enc_size'])
        self.i_n_val_emb = nn.Linear(args['node_enc_size'], args['node_enc_size'])
        self.i_n_att = nn.MultiheadAttention(args['node_enc_size'], num_heads=1)
        self.i_n_mix = nn.Linear(args['node_enc_size']*2, args['node_enc_size'])

        # Stopline-node attention
        self.s_n_query_emb = nn.Linear(args['node_enc_size'], args['node_enc_size'])
        self.s_n_key_emb = nn.Linear(args['node_enc_size'], args['node_enc_size'])
        self.s_n_val_emb = nn.Linear(args['node_enc_size'], args['node_enc_size'])
        self.s_n_att = nn.MultiheadAttention(args['node_enc_size'], num_heads=1)
        self.s_n_mix = nn.Linear(args['node_enc_size']*2, args['node_enc_size'])

        # Crosswalk-node attention
        self.c_n_query_emb = nn.Linear(args['node_enc_size'], args['node_enc_size'])
        self.c_n_key_emb = nn.Linear(args['node_enc_size'], args['node_enc_size'])
        self.c_n_val_emb = nn.Linear(args['node_enc_size'], args['node_enc_size'])
        self.c_n_att = nn.MultiheadAttention(args['node_enc_size'], num_heads=1)
        self.c_n_mix = nn.Linear(args['node_enc_size']*2, args['node_enc_size'])


        self.mix = nn.Linear(args['node_enc_size']*2, args['node_enc_size'])

        self.final_mix = nn.Linear(args['node_enc_size']*4, args['node_enc_size'])
        # Non-linearities
        self.leaky_relu = nn.LeakyReLU()

        # GAT layers
        self.gat = nn.ModuleList([GAT(args['node_enc_size'], args['node_enc_size'])
                                  for _ in range(args['num_gat_layers'])])

    def forward(self, inputs: Dict) -> Dict:
        """
        Forward pass for PGP encoder
        :param inputs: Dictionary with
            target_agent_representation: torch.Tensor, shape [batch_size, t_h, target_agent_feat_size]
            map_representation: Dict with
                'lane_node_feats': torch.Tensor, shape [batch_size, max_nodes, max_poses, node_feat_size]
                'lane_node_masks': torch.Tensor, shape [batch_size, max_nodes, max_poses, node_feat_size]

                (Optional)
                's_next': Edge look-up table pointing to destination node from source node
                'edge_type': Look-up table with edge type

            surrounding_agent_representation: Dict with
                'vehicles': torch.Tensor, shape [batch_size, max_vehicles, t_h, nbr_feat_size]
                'vehicle_masks': torch.Tensor, shape [batch_size, max_vehicles, t_h, nbr_feat_size]
                'pedestrians': torch.Tensor, shape [batch_size, max_peds, t_h, nbr_feat_size]
                'pedestrian_masks': torch.Tensor, shape [batch_size, max_peds, t_h, nbr_feat_size]
            agent_node_masks:  Dict with
                'vehicles': torch.Tensor, shape [batch_size, max_nodes, max_vehicles]
                'pedestrians': torch.Tensor, shape [batch_size, max_nodes, max_pedestrians]

            Optionally may also include the following if edges are defined for graph traversal
            'init_node': Initial node in the lane graph based on track history.
            'node_seq_gt': Ground truth node sequence for pre-training

        :return:
        """
        ############################ <Target Agent> ################################

        # Encode target agent
        target_agent_feats = inputs['target_agent_representation']
        target_agent_embedding = self.leaky_relu(self.target_agent_emb(target_agent_feats))
        _, target_agent_enc = self.target_agent_enc(target_agent_embedding)

        target_agent_enc = target_agent_enc.squeeze(0)
    

        # ############################ <Lane Nodes> ################################

        # Encode lane nodes
        lane_node_feats = inputs['map_representation']['lane_node_feats']
        lane_node_masks = inputs['map_representation']['lane_node_masks']
        lane_node_embedding = self.leaky_relu(self.node_emb(lane_node_feats))
        lane_node_enc = self.variable_size_gru_encode(lane_node_embedding, lane_node_masks, self.node_encoder)

        ############################ <Surrounding Agents > ################################

        # Encode surrounding agents
        nbr_vehicle_feats = inputs['surrounding_agent_representation']['vehicles']
        nbr_vehicle_masks = inputs['surrounding_agent_representation']['vehicle_masks']

        # nbr_vehicle_feats
        nbr_vehicle_feats = torch.cat((nbr_vehicle_feats, torch.zeros_like(nbr_vehicle_feats[:, :, :, 0:1])), dim=-1)
        # nbr_vehicle_masks = inputs['surrounding_agent_representation']['vehicle_masks']
        nbr_vehicle_embedding = self.leaky_relu(self.nbr_emb(nbr_vehicle_feats))
        nbr_vehicle_enc = self.variable_size_gru_encode(nbr_vehicle_embedding, nbr_vehicle_masks, self.nbr_enc)


        nbr_ped_feats = inputs['surrounding_agent_representation']['pedestrians']
        nbr_ped_masks = inputs['surrounding_agent_representation']['pedestrian_masks']
        nbr_ped_feats = torch.cat((nbr_ped_feats, torch.ones_like(nbr_ped_feats[:, :, :, 0:1])), dim=-1)
        nbr_ped_embedding = self.leaky_relu(self.nbr_emb(nbr_ped_feats))
        nbr_ped_enc = self.variable_size_gru_encode(nbr_ped_embedding, nbr_ped_masks, self.nbr_enc)

        # Agent-node attention

        nbr_encodings = torch.cat((nbr_vehicle_enc, nbr_ped_enc), dim=1)



        
        queries = self.query_emb(lane_node_enc).permute(1, 0, 2)
        keys = self.key_emb(nbr_encodings).permute(1, 0, 2)
        vals = self.val_emb(nbr_encodings).permute(1, 0, 2)


        a_n_attn_masks = torch.cat((inputs['agent_node_masks']['vehicles'],
                                inputs['agent_node_masks']['pedestrians']), dim=2)
        
        # print("a_n_attn_masks.shape --> ", a_n_attn_masks.shape)
        # print("-------------------")

        a_n_att_op, _ = self.a_n_att(queries, keys, vals, attn_mask=a_n_attn_masks)
        a_n_att_op = a_n_att_op.permute(1, 0, 2)

        # Concatenate with original node encodings and 1x1 conv
        a_node_mix = self.leaky_relu(self.a_n_mix(torch.cat((lane_node_enc, a_n_att_op), dim=2)))


        # Create an identity matrix of shape (164, 164)
        identity_matrix = torch.eye(lane_node_enc.shape[1], device=device)
        expanded_identity_matrix = identity_matrix.unsqueeze(0).expand(lane_node_enc.shape[0], -1, -1)
        
        


        intersection_masks = inputs['map_elements_node_masks']['intersections']
        any_zero = torch.any(intersection_masks == 0, dim=2)
        new_intersection_masks = torch.where(any_zero, torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
        new_intersection_masks = expanded_identity_matrix * new_intersection_masks.unsqueeze(2)

        # print("lane_node_enc.shape --> ", lane_node_enc.shape)
        # print("new_intersection_masks.shape --> ", new_intersection_masks.shape)

        i_n_queries = self.i_n_query_emb(lane_node_enc).permute(1, 0, 2)
        i_n_keys = self.i_n_key_emb(lane_node_enc).permute(1, 0, 2)
        i_n_vals = self.i_n_val_emb(lane_node_enc).permute(1, 0, 2)
        i_n_att_op, _ = self.i_n_att(i_n_queries, i_n_keys, i_n_vals, attn_mask=new_intersection_masks)
        i_n_att_op = i_n_att_op.permute(1, 0, 2)

        i_n_mix = self.leaky_relu(self.i_n_mix(torch.cat((lane_node_enc, i_n_att_op), dim=2)))


        stopline_masks = inputs['map_elements_node_masks']['stopline']
        any_zero = torch.any(stopline_masks == 0, dim=2)
        new_stopline_masks = torch.where(any_zero, torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
        new_stopline_masks = expanded_identity_matrix * new_stopline_masks.unsqueeze(2)

        s_n_queries = self.s_n_query_emb(lane_node_enc).permute(1, 0, 2)
        s_n_keys = self.s_n_key_emb(lane_node_enc).permute(1, 0, 2)
        s_n_vals = self.s_n_val_emb(lane_node_enc).permute(1, 0, 2)
        s_n_att_op, _ = self.s_n_att(s_n_queries, s_n_keys, s_n_vals, attn_mask=new_stopline_masks)
        s_n_att_op = s_n_att_op.permute(1, 0, 2)

        s_n_mix = self.leaky_relu(self.s_n_mix(torch.cat((lane_node_enc, s_n_att_op), dim=2)))


        crosswalk_masks = inputs['map_elements_node_masks']['ped_crossing']
        any_zero = torch.any(crosswalk_masks == 0, dim=2)
        new_crosswalk_masks = torch.where(any_zero, torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
        new_crosswalk_masks = expanded_identity_matrix * new_crosswalk_masks.unsqueeze(2)

        c_n_queries = self.c_n_query_emb(lane_node_enc).permute(1, 0, 2)
        c_n_keys = self.c_n_key_emb(lane_node_enc).permute(1, 0, 2)
        c_n_vals = self.c_n_val_emb(lane_node_enc).permute(1, 0, 2)
        c_n_att_op, _ = self.c_n_att(c_n_queries, c_n_keys, c_n_vals, attn_mask=new_crosswalk_masks)
        c_n_att_op = c_n_att_op.permute(1, 0, 2)

        c_n_mix = self.leaky_relu(self.c_n_mix(torch.cat((lane_node_enc, c_n_att_op), dim=2)))
        


        lane_node_enc = self.leaky_relu(self.mix(torch.cat((lane_node_enc, a_n_att_op), dim=2)))

        lane_node_enc = self.leaky_relu(self.final_mix(torch.cat((lane_node_enc, i_n_mix, s_n_mix, c_n_mix), dim=2)))


        # GAT layers
        adj_mat = self.build_adj_mat(inputs['map_representation']['s_next'], inputs['map_representation']['edge_type'])
        for gat_layer in self.gat:
            lane_node_enc += gat_layer(lane_node_enc, adj_mat)

        # Lane node masks
        lane_node_masks = ~lane_node_masks[:, :, :, 0].bool()
        lane_node_masks = lane_node_masks.any(dim=2)
        lane_node_masks = ~lane_node_masks
        lane_node_masks = lane_node_masks.float()

        target_agent_enc = target_agent_enc.unsqueeze(0)

        # Return encodings
        encodings = {'target_agent_encoding': target_agent_enc,
                     'surrounding_agent_encoding': nbr_encodings,
                     'context_encoding':lane_node_enc,
                     }

        # Pass on initial nodes and edge structure to aggregator if included in inputs
        if 'init_node' in inputs:
            encodings['init_node'] = inputs['init_node']
            encodings['node_seq_gt'] = inputs['node_seq_gt']
            encodings['s_next'] = inputs['map_representation']['s_next']
            encodings['edge_type'] = inputs['map_representation']['edge_type']

        return encodings

    @staticmethod
    def variable_size_gru_encode(feat_embedding: torch.Tensor, masks: torch.Tensor, gru: nn.GRU) -> torch.Tensor:
        """
        Returns GRU encoding for a batch of inputs where each sample in the batch is a set of a variable number
        of sequences, of variable lengths.
        """

        # Form a large batch of all sequences in the batch
        masks_for_batching = ~masks[:, :, :, 0].bool()
        masks_for_batching = masks_for_batching.any(dim=-1).unsqueeze(2).unsqueeze(3)
        feat_embedding_batched = torch.masked_select(feat_embedding, masks_for_batching)
        feat_embedding_batched = feat_embedding_batched.view(-1, feat_embedding.shape[2], feat_embedding.shape[3])

        # Pack padded sequences
        seq_lens = torch.sum(1 - masks[:, :, :, 0], dim=-1)
        seq_lens_batched = seq_lens[seq_lens != 0].cpu()
        if len(seq_lens_batched) != 0:
            feat_embedding_packed = pack_padded_sequence(feat_embedding_batched, seq_lens_batched,
                                                         batch_first=True, enforce_sorted=False)

            # Encode
            _, encoding_batched = gru(feat_embedding_packed)
            encoding_batched = encoding_batched.squeeze(0)

            # Scatter back to appropriate batch index
            masks_for_scattering = masks_for_batching.squeeze(3).repeat(1, 1, encoding_batched.shape[-1])
            encoding = torch.zeros(masks_for_scattering.shape, device=device)
            encoding = encoding.masked_scatter(masks_for_scattering, encoding_batched)

        else:
            batch_size = feat_embedding.shape[0]
            max_num = feat_embedding.shape[1]
            hidden_state_size = gru.hidden_size
            encoding = torch.zeros((batch_size, max_num, hidden_state_size), device=device)

        return encoding

    @staticmethod
    def build_adj_mat(s_next, edge_type):
        """
        Builds adjacency matrix for GAT layers.
        """
        batch_size = s_next.shape[0]
        max_nodes = s_next.shape[1]
        max_edges = s_next.shape[2]
        adj_mat = torch.diag(torch.ones(max_nodes, device=device)).unsqueeze(0).repeat(batch_size, 1, 1).bool()

        dummy_vals = torch.arange(max_nodes, device=device).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, max_edges)
        dummy_vals = dummy_vals.float()
        s_next[edge_type == 0] = dummy_vals[edge_type == 0]
        batch_indices = torch.arange(batch_size).unsqueeze(1).unsqueeze(2).repeat(1, max_nodes, max_edges)
        src_indices = torch.arange(max_nodes).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, max_edges)
        adj_mat[batch_indices[:, :, :-1], src_indices[:, :, :-1], s_next[:, :, :-1].long()] = True
        adj_mat = adj_mat | torch.transpose(adj_mat, 1, 2)

        return adj_mat


class GAT(nn.Module):
    """
    GAT layer for aggregating local context at each lane node. Uses scaled dot product attention using pytorch's
    multihead attention module.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initialize GAT layer.
        :param in_channels: size of node encodings
        :param out_channels: size of aggregated node encodings
        """
        super().__init__()
        self.query_emb = nn.Linear(in_channels, out_channels)
        self.key_emb = nn.Linear(in_channels, out_channels)
        self.val_emb = nn.Linear(in_channels, out_channels)
        self.att = nn.MultiheadAttention(out_channels, 1)

    def forward(self, node_encodings, adj_mat):
        """
        Forward pass for GAT layer
        :param node_encodings: Tensor of node encodings, shape [batch_size, max_nodes, node_enc_size]
        :param adj_mat: Bool tensor, adjacency matrix for edges, shape [batch_size, max_nodes, max_nodes]
        :return:
        """
        queries = self.query_emb(node_encodings.permute(1, 0, 2))
        keys = self.key_emb(node_encodings.permute(1, 0, 2))
        vals = self.val_emb(node_encodings.permute(1, 0, 2))
        att_op, _ = self.att(queries, keys, vals, attn_mask=~adj_mat)

        return att_op.permute(1, 0, 2)
