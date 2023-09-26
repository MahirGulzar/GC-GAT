import copy

from models.encoders.encoder import PredictionEncoder

from positional_encodings.torch_encodings import PositionalEncodingPermute1D, PositionalEncoding1D, PositionalEncoding2D, Summer



import numpy as np
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
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

        self.args = args

        dropout = 0.2

        # Target agent encoding
        self.target_agent_temporal_encoder_layer = TransformerEncoderLayer(d_model=args['target_agent_feat_size'] * 2, nhead=5)
        self.target_agent_temporal_encoder = TransformerEncoder(self.target_agent_temporal_encoder_layer, 2)
        self.dropout_target = nn.Dropout(dropout)
        self.target_agent_emb_enc = nn.Linear(args['target_agent_feat_size'] * 2, args['target_agent_enc_size'])

        self.target_agent_conv1d = nn.Conv1d(args['target_agent_feat_size'], args['target_agent_enc_size'], args['target_agent_enc_size'])

        # Surrounding agent encoding
        self.nbr_emb = nn.Linear(args['nbr_feat_size'] + 1, args['nbr_emb_size'])
        self.nbr_agent_temporal_encoder_layer = TransformerEncoderLayer(d_model=(args['nbr_feat_size'] + 1) * 2, nhead=6)
        self.nbr_agent_temporal_encoder = TransformerEncoder(self.nbr_agent_temporal_encoder_layer, 2)
        self.dropout_nbr = nn.Dropout(dropout)
        self.nbr_emb_enc = nn.Linear((args['nbr_feat_size'] + 1) * 2, args['nbr_enc_size'])

        # ReLU and dropout init
        self.relu = nn.ReLU()

        # nbr-nbr attention
        self.nbr_query_emb = nn.Linear(args['nbr_enc_size'], args['nbr_enc_size'])
        self.nbr_key_emb = nn.Linear(32, args['nbr_enc_size'])
        self.nbr_val_emb = nn.Linear(32, args['nbr_enc_size'])
        self.n_n_att = nn.MultiheadAttention(args['nbr_enc_size'], num_heads=1)
        # self.nbr_mix = nn.Linear(args['nbr_enc_size']*2, args['nbr_enc_size'])

        # Node encoders
        self.node_emb = nn.Linear(args['node_feat_size'], args['node_emb_size'])
        self.node_encoder = nn.GRU(args['node_emb_size'], args['node_enc_size'], batch_first=True)

        # # Agent-node attention
        # self.query_emb = nn.Linear(args['node_enc_size'], args['node_enc_size'])
        # self.key_emb = nn.Linear(32, args['node_enc_size'])
        # self.val_emb = nn.Linear(32, args['node_enc_size'])
        # self.a_n_att = nn.MultiheadAttention(args['node_enc_size'], num_heads=1)
        # self.mix = nn.Linear(args['node_enc_size']*2, args['node_enc_size'])

        # Agent-node attention ( actually surrounding agent attention with lane nodes )
        self.query_emb = nn.Linear(args['node_enc_size'], args['node_enc_size'])
        self.key_emb = nn.Linear(args['nbr_enc_size'], args['node_enc_size'])
        self.val_emb = nn.Linear(args['nbr_enc_size'], args['node_enc_size'])
        self.a_n_att = nn.MultiheadAttention(args['node_enc_size'], num_heads=1)
        self.mix = nn.Linear(args['node_enc_size']*2, args['node_enc_size'])

        # Target agent attention
        self.target_query_emb = nn.Linear(args['node_enc_size'], args['node_enc_size'])
        self.target_key_emb = nn.Linear(args['target_agent_enc_size'], args['node_enc_size'])
        self.target_val_emb = nn.Linear(args['target_agent_enc_size'], args['node_enc_size'])
        self.t_n_att = nn.MultiheadAttention(args['node_enc_size'], num_heads=1)
        self.target_mix = nn.Linear(args['node_enc_size']*2, args['node_enc_size'])


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

        #---------------------------------------------------------------------------------------
        # Encode target agent
        
        target_agent_feats = inputs['target_agent_representation']
        target_agent_pos_enc  = PositionalEncodingPermute1D(self.args['history_size'])
        # target_agent_feats = target_agent_pos_enc(target_agent_feats) + target_agent_feats
        target_agent_feats = torch.cat((target_agent_pos_enc(target_agent_feats), target_agent_feats), -1)

        target_agent_temporal_enc = self.target_agent_temporal_encoder(self.leaky_relu(target_agent_feats))
        target_agent_enc = self.dropout_target(self.leaky_relu(self.target_agent_emb_enc(target_agent_temporal_enc)))


        target_agent_enc = self.target_agent_conv1d(target_agent_enc)
        target_agent_enc = target_agent_enc.permute(2, 0, 1)
        target_agent_enc = target_agent_enc.squeeze(0)

        #---------------------------------------------------------------------------------------
        # Encode surrounding agents
        
        nbr_vehicle_feats = inputs['surrounding_agent_representation']['vehicles']
        nbr_vehicle_feats = torch.cat((nbr_vehicle_feats, torch.zeros_like(nbr_vehicle_feats[:, :, :, 0:1])), dim=-1)
        nbr_vehicle_masks = inputs['surrounding_agent_representation']['vehicle_masks']
        nbr_vehicle_enc = self.variable_size_transform_encode(nbr_vehicle_feats, nbr_vehicle_masks, self.nbr_agent_temporal_encoder)
        nbr_vehicle_enc = self.dropout_nbr(self.leaky_relu(self.nbr_emb_enc(nbr_vehicle_enc)))


        nbr_ped_feats = inputs['surrounding_agent_representation']['pedestrians']
        nbr_ped_feats = torch.cat((nbr_ped_feats, torch.ones_like(nbr_ped_feats[:, :, :, 0:1])), dim=-1)
        nbr_ped_masks = inputs['surrounding_agent_representation']['pedestrian_masks']


        nbr_pedestrian_enc = self.variable_size_transform_encode(nbr_ped_feats, nbr_ped_masks, self.nbr_agent_temporal_encoder)
        nbr_pedestrian_enc = self.dropout_nbr(self.leaky_relu(self.nbr_emb_enc(nbr_pedestrian_enc)))


        #---------------------------------------------------------------------------------------
        # Encode lane nodes
        lane_node_feats = inputs['map_representation']['lane_node_feats']
        lane_node_masks = inputs['map_representation']['lane_node_masks']
        lane_node_embedding = self.leaky_relu(self.node_emb(lane_node_feats))
        lane_node_enc = self.variable_size_gru_encode(lane_node_embedding, lane_node_masks, self.node_encoder)

        nbr_encodings = torch.cat((nbr_vehicle_enc, nbr_pedestrian_enc), dim=1)
        #---------------------------------------------------------------------------------------
        # nbr-nbr attention
        
        # queries = self.nbr_query_emb(nbr_encodings).permute(1, 0, 2)
        # keys = self.nbr_key_emb(nbr_encodings).permute(1, 0, 2)
        # vals = self.nbr_val_emb(nbr_encodings).permute(1, 0, 2)

        # n_n_att_op, _ = self.n_n_att(queries, keys, vals)
        # n_n_att_op = n_n_att_op.permute(1, 0, 2)


        #---------------------------------------------------------------------------------------

        # Target agent attention
        # print("lane_node_enc.shape --> ", lane_node_enc.shape)
        # print("nbr_encodings.shape --> ", nbr_encodings.shape)
        # print("target_agent_enc.shape --> ", target_agent_enc.shape)
        

        lane_node_queries = self.target_query_emb(lane_node_enc).permute(1, 0, 2)
        target_agent_node_keys = self.target_key_emb(target_agent_enc.unsqueeze(1)).permute(1, 0, 2)
        target_agent_node_vals = self.target_val_emb(target_agent_enc.unsqueeze(1)).permute(1, 0, 2)
        t_n_att_op, _ = self.t_n_att(lane_node_queries, target_agent_node_keys, target_agent_node_vals)
        t_n_att_op = t_n_att_op.permute(1, 0, 2)

        # print("t_n_att_op.shape --> ", t_n_att_op.shape)

        lane_target_enc = self.leaky_relu(self.target_mix(torch.cat((lane_node_enc, t_n_att_op), dim=2)))

        #---------------------------------------------------------------------------------------

        # Agent-node attention
        
        queries = self.query_emb(lane_node_enc).permute(1, 0, 2)
        keys = self.key_emb(nbr_encodings).permute(1, 0, 2)
        vals = self.val_emb(nbr_encodings).permute(1, 0, 2)
        attn_masks = torch.cat((inputs['agent_node_masks']['vehicles'],
                                inputs['agent_node_masks']['pedestrians']), dim=2)
        att_op, _ = self.a_n_att(queries, keys, vals, attn_mask=attn_masks)
        att_op = att_op.permute(1, 0, 2)

        # print("att_op.shape --> ", att_op.shape)

        # Concatenate with original node encodings and 1x1 conv
        lane_node_enc = self.leaky_relu(self.mix(torch.cat((lane_node_enc, att_op), dim=2)))

        lane_node_enc = lane_node_enc + lane_target_enc

        # print("lane_node_enc.shape --> ", lane_node_enc.shape)

        # print("after new attention lane_node_enc.shape --> ", lane_node_enc.shape)

        # print("--------------------------")


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

        # print("-------------------")

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
    def feature_masker(feat_embedding: torch.Tensor, masks: torch.Tensor, transform_encoder: nn.Module) -> torch.Tensor:
        """
        Returns an encoding for a batch of inputs where each sample in the batch is a set of a variable number
        of sequences, of variable lengths.
        """

        # Form a large batch of all sequences in the batch
        masks_for_batching = ~masks[:, :, :, 0].bool()
        masks_for_batching = masks_for_batching.any(dim=-1).unsqueeze(2).unsqueeze(3)
        feat_embedding_batched = torch.masked_select(feat_embedding, masks_for_batching)
        feat_embedding_batched = feat_embedding_batched.view(-1, feat_embedding.shape[2], feat_embedding.shape[3])

        # Encode
        encoding_batched = transform_encoder(feat_embedding_batched)
        encoding_batched = encoding_batched.squeeze(0)

        # Scatter back to appropriate batch index
        masks_for_scattering = masks_for_batching.squeeze(3).repeat(1, 1, encoding_batched.shape[-1])
        encoding = torch.zeros(masks_for_scattering.shape, device=device)
        encoding = encoding.masked_scatter(masks_for_scattering, encoding_batched)

        return encoding

    def variable_size_transform_encode(self, feat_embedding: torch.Tensor, masks: torch.Tensor, transform_encoder: nn.Module) -> torch.Tensor:
        """
        Returns an encoding for a batch of inputs where each sample in the batch is a set of a variable number
        of sequences, of variable lengths.
        """


        # Form a large batch of all sequences in the batch
        masks_for_batching = ~masks[:, :, :, 0].bool()
        masks_for_batching = masks_for_batching.any(dim=-1).unsqueeze(2).unsqueeze(3)
        feat_embedding_batched = torch.masked_select(feat_embedding, masks_for_batching)
        feat_embedding_batched = feat_embedding_batched.view(-1, feat_embedding.shape[2], feat_embedding.shape[3])

        # print("feat_embedding_batched.shape -->", feat_embedding_batched.shape)
        nbr_pos_enc = PositionalEncodingPermute1D(self.args['nbr_feat_size'])
        feat_embedding_batched = torch.cat((nbr_pos_enc(feat_embedding_batched), feat_embedding_batched), -1)
        # feat_embedding_batched = nbr_pos_enc(feat_embedding_batched) + feat_embedding_batched
        # print("feat_embedding_batched.shape -->", feat_embedding_batched.shape)

        # Encode
        encoding_batched = transform_encoder(self.leaky_relu(feat_embedding_batched))
        encoding_batched = encoding_batched.squeeze(0)

        # Scatter back to appropriate batch index
        masks_for_scattering = masks_for_batching.squeeze(3).repeat(1, 1, encoding_batched.shape[-1])
        encoding = torch.zeros(masks_for_scattering.shape, device=device)
        encoding = encoding.masked_scatter(masks_for_scattering, encoding_batched)

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

