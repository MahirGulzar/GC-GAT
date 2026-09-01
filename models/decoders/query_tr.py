from models.decoders.decoder import PredictionDecoder
import torch
import torch.nn as nn
from typing import Dict, Union
import torch.nn.functional as F
from models.ea_net_g import SocialCellGlobal, EAMLP
from models.decoders.utils import cluster_traj

# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QueryTr(PredictionDecoder):
    def __init__(self, args):
        super().__init__()


        self.agg_type = args['agg_type']
        self.num_samples = args['num_samples']
        self.op_len = args['op_len']
        self.lv_dim = args['lv_dim']
        self.hidden = nn.Linear(args['encoding_size'] + args['lv_dim'], args['hidden_size_pgp'])
        self.op_traj = nn.Linear(args['hidden_size_pgp'], args['op_len'] * 2)
        self.leaky_relu = nn.LeakyReLU()
        self.num_clusters = args['num_clusters']


        #--------------------------------------------------

        self.num_modes = args['num_modes']
        self.future_steps = args['op_len']
        self.hidden_size = args['hidden_size']
        self.min_scale = args['min_scale']

        self.decoder = nn.GRU(input_size=self.hidden_size,
                              hidden_size=self.hidden_size,
                              num_layers=1,
                              bias=True,
                              batch_first=False,
                              dropout=0,
                              bidirectional=False)

        # Laplace MDNs
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 2))
        self.scale = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 2))

        # k modes interaction
        self.lvm = nn.Sequential(
            nn.Linear(self.hidden_size + 8, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True))
        self.ea_net = SocialCellGlobal(self.hidden_size, self.hidden_size, self.future_steps, self.future_steps)
        self.aggr_embed = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True))
        

        # Learnable parameters for adjusting probabilities
        self.weight1 = nn.Parameter(torch.randn(32, 10))
        self.weight2 = nn.Parameter(torch.randn(32, 10))

    def forward(self, inputs: Union[Dict, torch.Tensor]) -> Dict:


        # agg_encoding = inputs['agg_encoding']

        # if self.agg_type == 'combined':
        #     agg_encoding = agg_encoding.unsqueeze(1).repeat(1, 32, 1)
        # else:
        #     if len(agg_encoding.shape) != 3 or agg_encoding.shape[1] != self.num_samples:
        #         raise Exception('Expected ' + str(self.num_samples) + 'encodings for each train/val data')

        # # Sample latent variable and concatenate with aggregated encoding
        # batch_size_pgp = agg_encoding.shape[0]
        # z_pgp = torch.randn(batch_size_pgp, 32, self.lv_dim, device=device)
        # agg_encoding = torch.cat((agg_encoding, z_pgp), dim=2)
        # h_pgp = self.leaky_relu(self.hidden(agg_encoding))

        # # Output trajectories
        # traj_pgp = self.op_traj(h_pgp)
        # traj_pgp = traj_pgp.reshape(batch_size_pgp, self.num_samples, self.op_len, 2)

        # Cluster
        # traj_clustered, probs = cluster_traj(self.num_clusters, traj_pgp)

        # print(traj_clustered.shape)

        # predictions = {'traj': traj_clustered, 'probs': probs}



        # -------------------------------------------------------
        mode_query_states = inputs['mode_query_states']  # [H, K x B, D]
        target = inputs['target']  # [1, K x B, D]
        pi = inputs['pi']  # [B, K]

        # print(mode_query_states.shape)
        # print(target.shape)
        # print(pi.shape)

        target_ = target.repeat(self.future_steps, 1, 1)  # [H, K x B, D]
        # print(target_.shape)
        z = torch.randn(target_.shape[0], target_.shape[1], 8, device=device)
        # print(z.shape)
        target_noise = self.lvm(torch.cat((target_, z), dim=-1))  # [H, K x B, D]
        target_noise = self.ea_net(target_noise)  # [H, K x B, D]
        aggr = self.aggr_embed(torch.cat((mode_query_states, target_noise), -1))  # [H, K x B, D]

        # Decoder
        out = self.decoder(aggr, target)[0]  # [H, K x B, D]
        out = out.transpose(0, 1)  # [K x B, H, D]

        # get loc and scale of Laplace Distribution
        loc = self.loc(out)  # [K x B, H, 2]
        scale = F.elu_(self.scale(out), alpha=1.0) + 1.0 + self.min_scale  # [K x B, H, 2]
        loc = loc.view(self.num_modes, -1, self.future_steps, 2)  # [K x B, H, 2]
        scale = scale.view(self.num_modes, -1, self.future_steps, 2)  # [K, B, H, 2]

        loc, scale = loc.transpose(0, 1), scale.transpose(0, 1)  # [B, K, H, 2]
        predictions = {'traj': loc, 'scale': scale, 'probs': pi}

        return predictions

        # #prob1 = torch.randn(32, 10)
        # prob2 = torch.randn(32, 10)

        # Expand probability tensors to match output shape
        # prob1_expanded = probs.unsqueeze(-1).unsqueeze(-1).expand_as(traj_clustered)
        # prob2_expanded = pi.unsqueeze(-1).unsqueeze(-1).expand_as(loc)

        # Compute weighted outputs
        # weighted_output1 = traj_clustered * prob1_expanded
        # weighted_output2 = loc * prob2_expanded


        # combined_probs = probs + pi
        # final_probs = combined_probs / combined_probs.sum(dim=1, keepdim=True)

        # final_probs_expanded = final_probs.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, traj_clustered.shape[2], traj_clustered.shape[3])
        
        # # Compute weighted outputs
        # weighted_output1 = traj_clustered * final_probs_expanded
        # weighted_output2 = loc * final_probs_expanded
        
        # # Sum the weighted outputs to get the final result
        # final_output = weighted_output1 + weighted_output2

        # # print(final_output.shape)
        # # print(final_probs.shape)  

        # # print(final_output)
        # # print(final_probs)  

        # predictions = {'traj': final_output, 'scale': scale, 'probs': final_probs}

        # if type(inputs) is dict:
        #     for key, val in inputs.items():
        #         if key != 'agg_encoding':
        #             predictions[key] = val

        # return predictions





