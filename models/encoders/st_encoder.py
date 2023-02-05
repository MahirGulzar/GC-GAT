import copy

from models.encoders.encoder import PredictionEncoder
from .multi_attention_forward import multi_head_attention_forward

from positional_encodings.torch_encodings import PositionalEncodingPermute1D, PositionalEncoding1D, PositionalEncoding2D, Summer



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from typing import Dict


# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class STEncoder(PredictionEncoder):

    def __init__(self, args: Dict):
        """
        GRU based encoder from PGP. Lane node features and agent histories encoded using GRUs.
        Additionally, agent-node attention layers infuse each node encoding with nearby agent context.
        Finally GAT layers aggregate local context at each node.

        args to include:

        target_agent_feat_size: int Size of target agent features
        target_agent_emb_size: int Size of target agent embedding
        taret_agent_enc_size: int Size of hidden state of target agent GRU encoder

        node_feat_size: int Size of lane node features
        node_emb_size: int Size of lane node embedding
        node_enc_size: int Size of hidden state of lane node GRU encoder

        nbr_feat_size: int Size of neighboring agent features
        nbr_enb_size: int Size of neighboring agent embeddings
        nbr_enc_size: int Size of hidden state of neighboring agent GRU encoders

        num_gat_layers: int Number of GAT layers to use.
        """

        super().__init__()

        # Target agent embeddings
        self.target_agent_emb = nn.Linear(args['target_agent_feat_size'], args['target_agent_emb_size'])
        self.target_agent_emb_enc = nn.Linear(args['target_agent_emb_size'], args['target_agent_enc_size'])

        # self.target_agent_enc = nn.GRU(args['target_agent_emb_size'], args['target_agent_enc_size'], batch_first=True)

        # Surrounding agent embeddings
        self.nbr_emb = nn.Linear(args['nbr_feat_size'] + 1, args['nbr_emb_size'])
        self.nbr_emb_enc = nn.Linear(args['nbr_emb_size'], args['nbr_enc_size'])

        # self.nbr_enc = nn.GRU(args['nbr_emb_size'], args['nbr_enc_size'], batch_first=True)

        self.target_agent_temporal_encoder_layer = TransformerEncoderLayer(d_model=args['target_agent_enc_size'], nhead=8)
        self.nbr_agent_temporal_encoder_layer = TransformerEncoderLayer(d_model=args['nbr_enc_size'], nhead=8)


        # Independent Temporal Encoders
        self.target_agent_temporal_encoder = TransformerEncoder(self.target_agent_temporal_encoder_layer, 1)
        self.nbr_agent_temporal_encoder = TransformerEncoder(self.nbr_agent_temporal_encoder_layer, 1)


        self.target_agent_conv1d = nn.Conv1d(args['target_agent_feat_size'], args['target_agent_enc_size'], args['target_agent_enc_size'])


        nhid = 2048  # the dimension of the feedforward network model in TransformerEncoder
        nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead = 8  # the number of heads in the multihead-attention models
        dropout = 0.1  # the dropout value

        # Independent Spatial Encoders
        # TODO: Replace these with TransformModel that uses masks

        target_agent_spatial_layers = TransformerEncoderLayer(args['target_agent_enc_size'], nhead, nhid, dropout)
        nbr_agent_spatial_layers = TransformerEncoderLayer(args['nbr_enc_size'], nhead, nhid, dropout)

        self.target_agent_spatial_encoder = TransformerEncoder(target_agent_spatial_layers, nlayers)
        self.nbr_agent_spatial_encoder = TransformerEncoder(nbr_agent_spatial_layers, nlayers)



        self.target_agent_fusion = nn.Linear(args['target_agent_enc_size'] * 2, 32)
        self.nbr_agent_fusion = nn.Linear(args['nbr_enc_size'] * 2, 32)
        self.agent_nbr_fusion = nn.Linear(args['target_agent_enc_size'] + args['nbr_enc_size'], 32)



        # Fused encoder

        # self.post_temporal_encoder_layer = TransformerEncoderLayer(d_model=32, nhead=8)

        # self.post_temporal_encoder = TransformerEncoder(self.post_temporal_encoder_layer, 1)
        # self.post_spatial_encoder = TransformerModel(32, nhead=nhead, nhid=nhid, nlayers=nlayers, dropout=dropout)


        # ReLU and dropout init
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(dropout)
        self.dropout_in2 = nn.Dropout(dropout)

        # -------------------------------------------------------------------------------------------

        # set parameters for network architecture
        # self.embedding_size = [32]
        # self.output_size = 2
        # self.dropout_prob = dropout_prob
        # self.args = args

        # self.temporal_encoder_layer = TransformerEncoderLayer(d_model=32, nhead=8)

        # emsize = 32  # embedding dimension
        # nhid = 2048  # the dimension of the feedforward network model in TransformerEncoder
        # nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        # nhead = 8  # the number of heads in the multihead-attention models
        # dropout = 0.1  # the dropout value

        # self.spatial_encoder_1 = TransformerModel(emsize, nhead, nhid, nlayers, dropout)
        # self.spatial_encoder_2 = TransformerModel(emsize, nhead, nhid, nlayers, dropout)

        # self.temporal_encoder_1 = TransformerEncoder(self.temporal_encoder_layer, 1)
        # self.temporal_encoder_2 = TransformerEncoder(self.temporal_encoder_layer, 1)


        # ----------------------------------------------------------------------------------------------


        # Node encoders
        self.node_emb = nn.Linear(args['node_feat_size'], args['node_emb_size'])
        self.node_encoder = nn.GRU(args['node_emb_size'], args['node_enc_size'], batch_first=True)

        # Agent-node attention
        self.query_emb = nn.Linear(args['node_enc_size'], args['node_enc_size'])
        self.key_emb = nn.Linear(32, args['node_enc_size'])
        self.val_emb = nn.Linear(32, args['node_enc_size'])
        self.a_n_att = nn.MultiheadAttention(args['node_enc_size'], num_heads=1)
        self.mix = nn.Linear(args['node_enc_size']*2, args['node_enc_size'])

        # Non-linearities
        self.leaky_relu = nn.LeakyReLU()

        # GAT layers
        self.gat = nn.ModuleList([GAT(args['node_enc_size'], args['node_enc_size'])
                                  for _ in range(args['num_gat_layers'])])


        # ---------------------------------------------------------------------------------------------

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

        # Encode target agent

        target_agent_feats = inputs['target_agent_representation']

        print(target_agent_feats[0][1])
        print(inputs['surrounding_agent_representation']['vehicles'][0][1])
        # print(inputs['surrounding_agent_representation']['vehicles'].shape)

        target_agent_emb = self.dropout_in(self.leaky_relu(self.target_agent_emb(target_agent_feats)))

        

        

        # target_agent_pos_enc = PositionalEncodingPermute1D()

        # print("target_agent_emb.shape --> ", target_agent_emb.shape)
        target_agent_emb = self.target_agent_emb_enc(target_agent_emb)

        # print(target_agent_emb.shape)
        # print("target_agent_emb_enc.shape --> ", target_agent_emb.shape)

        target_agent_pos_enc_summer = Summer(PositionalEncoding1D(target_agent_emb.shape[len(target_agent_emb.shape)-1]))
        target_agent_emb = target_agent_pos_enc_summer(target_agent_emb)
        
        # print("positional encoding")
        # print(target_agent_emb.shape)
        # print(target_agent_emb)

        # print("------------------------")


        target_agent_temporal_enc = self.target_agent_temporal_encoder(target_agent_emb)
        # target_agent_spatial_enc = self.target_agent_spatial_encoder(target_agent_emb)

        # print("target_agent_temporal_enc.shape --> ", target_agent_temporal_enc.shape)
        # print("target_agent_spatial_enc.shape --> ", target_agent_spatial_enc.shape)

        ##target_agent_temporal_enc = target_agent_temporal_enc.squeeze(0)
        ##target_agent_spatial_enc = target_agent_spatial_enc.squeeze(0)

        # print("target_agent_temporal_enc.shape squeeze(0) --> ", target_agent_temporal_enc.shape)
        # print("target_agent_spatial_enc.shape squeeze(0)--> ", target_agent_spatial_enc.shape)

        ### target_agent_concat_enc = torch.cat((target_agent_temporal_enc, target_agent_spatial_enc), dim=1)
        # print(target_agent_concat_enc.shape)
        # target_agent_enc = target_agent_concat_enc.squeeze(0).permute(1, 0, 2)


        target_agent_enc = self.target_agent_conv1d(target_agent_temporal_enc)
        # print("target_agent_enc.shape) --> ", target_agent_enc.shape)

        target_agent_enc = target_agent_enc.permute(2, 0, 1)
        # print("target_agent_enc.permute(2, 0, 1)) --> ", target_agent_enc.shape)

        target_agent_enc = target_agent_enc.squeeze(0)
        # print("target_agent_enc.squeeze(0)) --> ", target_agent_enc.shape)

        # target_agent_enc = self.target_agent_fusion(target_agent_concat_enc)
        # print("target_agent_enc.shape", target_agent_enc.shape)
        # print("target_agent_enc.permute(1, 0, 2).shape", target_agent_enc.permute(1, 0, 2).shape)
        


        # Encode surrounding agents
        
        nbr_vehicle_feats = inputs['surrounding_agent_representation']['vehicles']
        print("nbr_vehicle_feats.shape -->", nbr_vehicle_feats.shape)
        print(nbr_vehicle_feats[0][0])
        nbr_vehicle_feats = torch.cat((nbr_vehicle_feats, torch.zeros_like(nbr_vehicle_feats[:, :, :, 0:1])), dim=-1)
        print("nbr_vehicle_feats.shape -->", nbr_vehicle_feats.shape)
        nbr_vehicle_masks = inputs['surrounding_agent_representation']['vehicle_masks']
        print("nbr_vehicle_masks.shape -->", nbr_vehicle_masks.shape)
        print(nbr_vehicle_masks[0][0])

        nbr_vehicle_emb = self.dropout_in2(self.leaky_relu(self.nbr_emb(nbr_vehicle_feats)))
        nbr_vehicle_emb = self.nbr_emb_enc(nbr_vehicle_emb)

        print("nbr_vehicle_emb.shape --> ", nbr_vehicle_emb.shape)

        nbr_vehicle_emb_summer = Summer(PositionalEncoding2D(nbr_vehicle_emb.shape[len(nbr_vehicle_emb.shape)-1]))
        nbr_vehicle_emb = nbr_vehicle_emb_summer(nbr_vehicle_emb)


        nbr_temporal_vehicle_enc = self.variable_size_transform_encode(nbr_vehicle_emb, nbr_vehicle_masks, self.nbr_agent_temporal_encoder)
        nbr_spatial_vehicle_enc = self.variable_size_transform_encode(nbr_vehicle_emb, nbr_vehicle_masks, self.nbr_agent_spatial_encoder)

        # print("nbr_temporal_vehicle_enc.shape --> ", nbr_temporal_vehicle_enc.shape)
        # print("nbr_spatial_vehicle_enc.shape --> ", nbr_spatial_vehicle_enc.shape)

        nbr_vehicle_concat_enc = torch.cat((nbr_temporal_vehicle_enc, nbr_spatial_vehicle_enc), dim=2)
        nbr_vehicle_enc = self.nbr_agent_fusion(nbr_vehicle_concat_enc)

        nbr_ped_feats = inputs['surrounding_agent_representation']['pedestrians']
        nbr_ped_feats = torch.cat((nbr_ped_feats, torch.ones_like(nbr_ped_feats[:, :, :, 0:1])), dim=-1)
        nbr_ped_masks = inputs['surrounding_agent_representation']['pedestrian_masks']


        nbr_pedestrian_emb = self.dropout_in2(self.leaky_relu(self.nbr_emb(nbr_ped_feats)))
        nbr_pedestrian_emb = self.nbr_emb_enc(nbr_pedestrian_emb)

        nbr_pedestrian_emb_summer = Summer(PositionalEncoding2D(nbr_pedestrian_emb.shape[len(nbr_pedestrian_emb.shape)-1]))
        nbr_pedestrian_emb = nbr_pedestrian_emb_summer(nbr_pedestrian_emb)

        nbr_temporal_pedestrian_enc = self.variable_size_transform_encode(nbr_pedestrian_emb, nbr_ped_masks, self.nbr_agent_temporal_encoder)
        nbr_spatial_pedestrian_enc = self.variable_size_transform_encode(nbr_pedestrian_emb, nbr_ped_masks, self.nbr_agent_spatial_encoder)

        # print("nbr_temporal_pedestrian_enc.shape --> ", nbr_temporal_pedestrian_enc.shape)
        # print("nbr_spatial_pedestrian_enc.shape --> ", nbr_spatial_pedestrian_enc.shape)


        nbr_pedestrian_concat_enc = torch.cat((nbr_temporal_pedestrian_enc, nbr_spatial_pedestrian_enc), dim=2)
        nbr_pedestrian_enc = self.nbr_agent_fusion(nbr_pedestrian_concat_enc)


        
        #---------------------------------------------------------------------------------------
        # Encode target agent
        # target_agent_feats = inputs['target_agent_representation']
        # target_agent_embedding = self.leaky_relu(self.target_agent_emb(target_agent_feats))
        # _, target_agent_enc = self.target_agent_enc(target_agent_embedding)
        # target_agent_enc = target_agent_enc.squeeze(0)

        # Encode lane nodes
        lane_node_feats = inputs['map_representation']['lane_node_feats']
        lane_node_masks = inputs['map_representation']['lane_node_masks']
        lane_node_embedding = self.leaky_relu(self.node_emb(lane_node_feats))
        lane_node_enc = self.variable_size_gru_encode(lane_node_embedding, lane_node_masks, self.node_encoder)

        # Encode surrounding agents
        # nbr_vehicle_feats = inputs['surrounding_agent_representation']['vehicles']
        # nbr_vehicle_feats = torch.cat((nbr_vehicle_feats, torch.zeros_like(nbr_vehicle_feats[:, :, :, 0:1])), dim=-1)
        # nbr_vehicle_masks = inputs['surrounding_agent_representation']['vehicle_masks']
        # nbr_vehicle_embedding = self.leaky_relu(self.nbr_emb(nbr_vehicle_feats))
        # nbr_vehicle_enc = self.variable_size_gru_encode(nbr_vehicle_embedding, nbr_vehicle_masks, self.nbr_enc)
        # nbr_ped_feats = inputs['surrounding_agent_representation']['pedestrians']
        # nbr_ped_feats = torch.cat((nbr_ped_feats, torch.ones_like(nbr_ped_feats[:, :, :, 0:1])), dim=-1)
        # nbr_ped_masks = inputs['surrounding_agent_representation']['pedestrian_masks']
        # nbr_ped_embedding = self.leaky_relu(self.nbr_emb(nbr_ped_feats))
        # nbr_ped_enc = self.variable_size_gru_encode(nbr_ped_embedding, nbr_ped_masks, self.nbr_enc)

        

        # print("nbr_vehicle_enc.shape", nbr_vehicle_enc.shape)
        # print("nbr_pedestrian_enc.shape", nbr_pedestrian_enc.shape)


        # Agent-node attention
        nbr_encodings = torch.cat((nbr_vehicle_enc, nbr_pedestrian_enc), dim=1)
        queries = self.query_emb(lane_node_enc).permute(1, 0, 2)
        keys = self.key_emb(nbr_encodings).permute(1, 0, 2)
        vals = self.val_emb(nbr_encodings).permute(1, 0, 2)
        attn_masks = torch.cat((inputs['agent_node_masks']['vehicles'],
                                inputs['agent_node_masks']['pedestrians']), dim=2)
        att_op, _ = self.a_n_att(queries, keys, vals, attn_mask=attn_masks)
        att_op = att_op.permute(1, 0, 2)

        # Concatenate with original node encodings and 1x1 conv
        lane_node_enc = self.leaky_relu(self.mix(torch.cat((lane_node_enc, att_op), dim=2)))

        # GAT layers
        adj_mat = self.build_adj_mat(inputs['map_representation']['s_next'], inputs['map_representation']['edge_type'])
        for gat_layer in self.gat:
            lane_node_enc += gat_layer(lane_node_enc, adj_mat)

        # Lane node masks
        lane_node_masks = ~lane_node_masks[:, :, :, 0].bool()
        lane_node_masks = lane_node_masks.any(dim=2)
        lane_node_masks = ~lane_node_masks
        lane_node_masks = lane_node_masks.float()

        # Return encodings
        encodings = {'target_agent_encoding': target_agent_enc,
                     'context_encoding': {'combined': lane_node_enc,
                                          'combined_masks': lane_node_masks,
                                          'map': None,
                                          'vehicles': None,
                                          'pedestrians': None,
                                          'map_masks': None,
                                          'vehicle_masks': None,
                                          'pedestrian_masks': None
                                          },
                     }

        # Pass on initial nodes and edge structure to aggregator if included in inputs
        if 'init_node' in inputs:
            encodings['init_node'] = inputs['init_node']
            encodings['node_seq_gt'] = inputs['node_seq_gt']
            encodings['s_next'] = inputs['map_representation']['s_next']
            encodings['edge_type'] = inputs['map_representation']['edge_type']

        print("-------------------")

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
    def variable_size_transform_encode(feat_embedding: torch.Tensor, masks: torch.Tensor, transform_encoder: nn.Module) -> torch.Tensor:
        """
        Returns an encoding for a batch of inputs where each sample in the batch is a set of a variable number
        of sequences, of variable lengths.
        """

        # Form a large batch of all sequences in the batch
        masks_for_batching = ~masks[:, :, :, 0].bool()
        print("masks_for_batching.shape", masks_for_batching.shape)
        print(masks_for_batching[0][0])
        masks_for_batching = masks_for_batching.any(dim=-1).unsqueeze(2).unsqueeze(3)
        print("masks_for_batching.shape after unsqueeze", masks_for_batching.shape)
        print(masks_for_batching[0][0])
        print("feat_embedding.shape", feat_embedding.shape)
        print(feat_embedding[0][0])
        feat_embedding_batched = torch.masked_select(feat_embedding, masks_for_batching)
        print("feat_embedding_batched.shape after masking", feat_embedding_batched.shape)
        print(feat_embedding_batched[0])
        feat_embedding_batched = feat_embedding_batched.view(-1, feat_embedding.shape[2], feat_embedding.shape[3])

        print("feat_embedding_batched.shape after masking and viewing", feat_embedding_batched.shape)
        print(feat_embedding_batched[0])
        print("oooooooooooooooooooooooooooooooooooooooooooooooooo")

        # # Pack padded sequences
        # seq_lens = torch.sum(1 - masks[:, :, :, 0], dim=-1)
        # seq_lens_batched = seq_lens[seq_lens != 0].cpu()



        # feat_embedding_packed = pack_padded_sequence(feat_embedding_batched, seq_lens_batched,
        #                                                  batch_first=True, enforce_sorted=False)

        # Encode
        encoding_batched = transform_encoder(feat_embedding_batched)
        encoding_batched = encoding_batched.squeeze(0)

        # Scatter back to appropriate batch index
        masks_for_scattering = masks_for_batching.squeeze(3).repeat(1, 1, encoding_batched.shape[-1])
        encoding = torch.zeros(masks_for_scattering.shape, device=device)
        encoding = encoding.masked_scatter(masks_for_scattering, encoding_batched)


        # if len(seq_lens_batched) != 0:
        #     feat_embedding_packed = pack_padded_sequence(feat_embedding_batched, seq_lens_batched,
        #                                                  batch_first=True, enforce_sorted=False)

        #     # Encode
        #     encoding_batched = transform_encoder(feat_embedding_packed)
        #     encoding_batched = encoding_batched.squeeze(0)

        #     # Scatter back to appropriate batch index
        #     masks_for_scattering = masks_for_batching.squeeze(3).repeat(1, 1, encoding_batched.shape[-1])
        #     encoding = torch.zeros(masks_for_scattering.shape, device=device)
        #     encoding = encoding.masked_scatter(masks_for_scattering, encoding_batched)

        # else:
        #     batch_size = feat_embedding.shape[0]
        #     max_num = feat_embedding.shape[1]
        #     hidden_state_size = transform_encoder.hidden_size
        #     encoding = torch.zeros((batch_size, max_num, hidden_state_size), device=device)

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



def get_noise(shape, noise_type):
    if noise_type == "gaussian":
        return torch.randn(shape).cuda()
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



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


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0, activation="relu", batch_first=True):
        super(TransformerEncoderLayer, self).__init__()
        # self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))

        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn


class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequnce to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        atts = []

        for i in range(self.num_layers):
            output, attn = self.layers[i](output, src_mask=mask,
                                          src_key_padding_mask=src_key_padding_mask)
            atts.append(attn)
        if self.norm:
            output = self.norm(output)

        return output


class TransformerModel(nn.Module):

    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp

    def forward(self, src, mask):
        n_mask = mask + torch.eye(mask.shape[0], mask.shape[0]).cuda()
        n_mask = n_mask.float().masked_fill(n_mask == 0., float(-1e20)).masked_fill(n_mask == 1., float(0.0))
        output = self.transformer_encoder(src, mask=n_mask)

        return output


# class STAR(torch.nn.Module):

#     def __init__(self, args, dropout_prob=0):
#         super(STAR, self).__init__()

#         # set parameters for network architecture
#         self.embedding_size = [32]
#         self.output_size = 2
#         self.dropout_prob = dropout_prob
#         self.args = args

#         self.temporal_encoder_layer = TransformerEncoderLayer(d_model=32, nhead=8)

#         emsize = 32  # embedding dimension
#         nhid = 2048  # the dimension of the feedforward network model in TransformerEncoder
#         nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
#         nhead = 8  # the number of heads in the multihead-attention models
#         dropout = 0.1  # the dropout value

#         self.spatial_encoder_1 = TransformerModel(emsize, nhead, nhid, nlayers, dropout)
#         self.spatial_encoder_2 = TransformerModel(emsize, nhead, nhid, nlayers, dropout)

#         self.temporal_encoder_1 = TransformerEncoder(self.temporal_encoder_layer, 1)
#         self.temporal_encoder_2 = TransformerEncoder(self.temporal_encoder_layer, 1)

#         # Linear layer to map input to embedding
#         self.input_embedding_layer_temporal = nn.Linear(2, 32)
#         self.input_embedding_layer_spatial = nn.Linear(2, 32)

#         # Linear layer to output and fusion
#         self.output_layer = nn.Linear(48, 2)
#         self.fusion_layer = nn.Linear(64, 32)

#         # ReLU and dropout init
#         self.relu = nn.ReLU()
#         self.dropout_in = nn.Dropout(self.dropout_prob)
#         self.dropout_in2 = nn.Dropout(self.dropout_prob)

#     def get_st_ed(self, batch_num):
#         """

#         :param batch_num: contains number of pedestrians in different scenes for a batch
#         :type batch_num: list
#         :return: st_ed: list of tuple contains start index and end index of pedestrians in different scenes
#         :rtype: list
#         """
#         cumsum = torch.cumsum(batch_num, dim=0)
#         st_ed = []
#         for idx in range(1, cumsum.shape[0]):
#             st_ed.append((int(cumsum[idx - 1]), int(cumsum[idx])))

#         st_ed.insert(0, (0, int(cumsum[0])))

#         return st_ed

#     def get_node_index(self, seq_list):
#         """

#         :param seq_list: mask indicates whether pedestrain exists
#         :type seq_list: numpy array [F, N], F: number of frames. N: Number of pedestrians (a mask to indicate whether
#                                                                                             the pedestrian exists)
#         :return: All the pedestrians who exist from the beginning to current frame
#         :rtype: numpy array
#         """
#         for idx, framenum in enumerate(seq_list):

#             if idx == 0:
#                 node_indices = framenum > 0
#             else:
#                 node_indices *= (framenum > 0)

#         return node_indices

#     def update_batch_pednum(self, batch_pednum, ped_list):
#         """

#         :param batch_pednum: batch_num: contains number of pedestrians in different scenes for a batch
#         :type list
#         :param ped_list: mask indicates whether the pedestrian exists through the time window to current frame
#         :type tensor
#         :return: batch_pednum: contains number of pedestrians in different scenes for a batch after removing pedestrian who disappeared
#         :rtype: list
#         """
#         updated_batch_pednum_ = copy.deepcopy(batch_pednum).cpu().numpy()
#         updated_batch_pednum = copy.deepcopy(batch_pednum)

#         cumsum = np.cumsum(updated_batch_pednum_)
#         new_ped = copy.deepcopy(ped_list).cpu().numpy()

#         for idx, num in enumerate(cumsum):
#             num = int(num)
#             if idx == 0:
#                 updated_batch_pednum[idx] = len(np.where(new_ped[0:num] == 1)[0])
#             else:
#                 updated_batch_pednum[idx] = len(np.where(new_ped[int(cumsum[idx - 1]):num] == 1)[0])

#         return updated_batch_pednum

#     def mean_normalize_abs_input(self, node_abs, st_ed):
#         """

#         :param node_abs: Absolute coordinates of pedestrians
#         :type Tensor
#         :param st_ed: list of tuple indicates the indices of pedestrians belonging to the same scene
#         :type List of tupule
#         :return: node_abs: Normalized absolute coordinates of pedestrians
#         :rtype: Tensor
#         """
#         node_abs = node_abs.permute(1, 0, 2)
#         for st, ed in st_ed:
#             mean_x = torch.mean(node_abs[st:ed, :, 0])
#             mean_y = torch.mean(node_abs[st:ed, :, 1])

#             node_abs[st:ed, :, 0] = (node_abs[st:ed, :, 0] - mean_x)
#             node_abs[st:ed, :, 1] = (node_abs[st:ed, :, 1] - mean_y)

#         return node_abs.permute(1, 0, 2)

#     def forward(self, inputs, iftest=False):

#         nodes_abs, nodes_norm, shift_value, seq_list, nei_lists, nei_num, batch_pednum = inputs
#         num_Ped = nodes_norm.shape[1]

#         outputs = torch.zeros(nodes_norm.shape[0], num_Ped, 2).cuda()
#         GM = torch.zeros(nodes_norm.shape[0], num_Ped, 32).cuda()

#         noise = get_noise((1, 16), 'gaussian')

#         for framenum in range(self.args.seq_length - 1):

#             if framenum >= self.args.obs_length and iftest:

#                 node_index = self.get_node_index(seq_list[:self.args.obs_length])
#                 updated_batch_pednum = self.update_batch_pednum(batch_pednum, node_index)
#                 st_ed = self.get_st_ed(updated_batch_pednum)

#                 nodes_current = outputs[self.args.obs_length - 1:framenum, node_index]
#                 nodes_current = torch.cat((nodes_norm[:self.args.obs_length, node_index], nodes_current))
#                 node_abs_base = nodes_abs[:self.args.obs_length, node_index]
#                 node_abs_pred = shift_value[self.args.obs_length:framenum + 1, node_index] + outputs[
#                                                                                            self.args.obs_length - 1:framenum,
#                                                                                            node_index]
#                 node_abs = torch.cat((node_abs_base, node_abs_pred), dim=0)
#                 # We normalize the absolute coordinates using the mean value in the same scene
#                 node_abs = self.mean_normalize_abs_input(node_abs, st_ed)

#             else:
#                 node_index = self.get_node_index(seq_list[:framenum + 1])
#                 nei_list = nei_lists[framenum, node_index, :]
#                 nei_list = nei_list[:, node_index]
#                 updated_batch_pednum = self.update_batch_pednum(batch_pednum, node_index)
#                 st_ed = self.get_st_ed(updated_batch_pednum)
#                 nodes_current = nodes_norm[:framenum + 1, node_index]
#                 # We normalize the absolute coordinates using the mean value in the same scene
#                 node_abs = self.mean_normalize_abs_input(nodes_abs[:framenum + 1, node_index], st_ed)

#             # Input Embedding
#             if framenum == 0:
#                 temporal_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_temporal(nodes_current)))
#             else:
#                 temporal_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_temporal(nodes_current)))
#                 temporal_input_embedded[:framenum] = GM[:framenum, node_index]

#             spatial_input_embedded_ = self.dropout_in2(self.relu(self.input_embedding_layer_spatial(node_abs)))

#             spatial_input_embedded = self.spatial_encoder_1(spatial_input_embedded_[-1].unsqueeze(1), nei_list)

#             spatial_input_embedded = spatial_input_embedded.permute(1, 0, 2)[-1]
#             temporal_input_embedded_last = self.temporal_encoder_1(temporal_input_embedded)[-1]

#             temporal_input_embedded = temporal_input_embedded[:-1]

#             fusion_feat = torch.cat((temporal_input_embedded_last, spatial_input_embedded), dim=1)
#             fusion_feat = self.fusion_layer(fusion_feat)

#             spatial_input_embedded = self.spatial_encoder_2(fusion_feat.unsqueeze(1), nei_list)
#             spatial_input_embedded = spatial_input_embedded.permute(1, 0, 2)

#             temporal_input_embedded = torch.cat((temporal_input_embedded, spatial_input_embedded), dim=0)
#             temporal_input_embedded = self.temporal_encoder_2(temporal_input_embedded)[-1]

#             noise_to_cat = noise.repeat(temporal_input_embedded.shape[0], 1)
#             temporal_input_embedded_wnoise = torch.cat((temporal_input_embedded, noise_to_cat), dim=1)
#             outputs_current = self.output_layer(temporal_input_embedded_wnoise)
#             outputs[framenum, node_index] = outputs_current
#             GM[framenum, node_index] = temporal_input_embedded

#         return outputs
