from models.decoders.decoder import PredictionDecoder
import torch
import torch.nn as nn
from typing import Dict, Union
from models.decoders.utils import cluster_traj


# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Future(PredictionDecoder):

    def __init__(self, args):
        """
        Latent variable conditioned decoder.

        args to include:
        agg_type: 'combined' or 'sample_specific'. Whether we have a single aggregated context vector or sample-specific
        num_samples: int Number of trajectories to sample
        op_len: int Length of predicted trajectories
        lv_dim: int Dimension of latent variable
        encoding_size: int Dimension of encoded scene + agent context
        hidden_size: int Size of output mlp hidden layer
        num_clusters: int Number of final clustered trajectories to output

        """
        super().__init__()
        self.agg_type = args['agg_type']
        self.num_samples = args['num_samples']
        self.num_modes = args['num_modes']
        self.op_len = args['op_len']
        self.lv_dim = args['lv_dim']
        self.hidden = nn.Linear(args['encoding_size'], args['hidden_size'])
        self.op_traj = nn.Linear(3200, args['op_len'] * 2 * self.num_modes)
        self.prob_op = nn.Linear(3200, self.num_modes)

        self.leaky_relu = nn.LeakyReLU()
        # self.num_clusters = args['num_clusters']
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs: Union[Dict, torch.Tensor]) -> Dict:
        """
        Forward pass for latent variable model.

        :param inputs: aggregated context encoding,
         shape for combined encoding: [batch_size, encoding_size]
         shape if sample specific encoding: [batch_size, num_samples, encoding_size]
        :return: predictions
        """
        agg_encoding = inputs['agg_encoding']
        preds = inputs['preds']

        # h = self.leaky_relu(self.hidden(agg_encoding))
        # h = torch.reshape(h, (h.shape[0], -1))

        # batch_size = h.shape[0]
        # traj = self.op_traj(h)

        # probs = self.log_softmax(self.prob_op(h))
        # traj = traj.reshape(batch_size, self.num_modes, self.op_len, 2)
        # probs = probs.squeeze(dim=-1)

        
        # ----------------------------------------

        # # Output trajectories
        # traj = self.op_traj(h)
        # traj = traj.reshape(batch_size, self.num_samples, self.op_len, 2)

        # # Cluster
        # traj_clustered, probs = cluster_traj(self.num_clusters, traj)

        

        preds = preds.unsqueeze(1).repeat(1, 10, 1, 1)
        # print("preds.shape: ", preds.shape)

        # z = torch.randn(preds.shape).to(device)

        # preds = preds + z
        probs = torch.ones(preds.shape[0], preds.shape[1]).to(device)
        # print("probs.shape: ", probs.shape)

        predictions = {'traj': preds, 'probs': probs}

        if type(inputs) is dict:
            for key, val in inputs.items():
                if key != 'agg_encoding':
                    predictions[key] = val

        return predictions
