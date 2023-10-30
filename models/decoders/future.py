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
        self.op_len = args['op_len']
        self.lv_dim = args['lv_dim']
        self.hidden = nn.Linear(args['encoding_size'] + args['lv_dim'], args['hidden_size'])
        self.hidden_2 = nn.Linear(32, args['hidden_size'])
        self.hidden_3 = nn.Linear(2000, 1000)
        self.op_traj = nn.Linear(args['hidden_size'], args['op_len'] * 2)
        self.leaky_relu = nn.LeakyReLU()
        self.num_clusters = args['num_clusters']


    def forward(self, inputs: Union[Dict, torch.Tensor]) -> Dict:
        """
        Forward pass for latent variable model.

        :param inputs: aggregated context encoding,
         shape for combined encoding: [batch_size, encoding_size]
         shape if sample specific encoding: [batch_size, num_samples, encoding_size]
        :return: predictions
        """

        if type(inputs) is torch.Tensor:
            agg_encoding = inputs
        else:
            agg_encoding = inputs['agg_encoding']

        if self.agg_type == 'combined':
            agg_encoding = agg_encoding.unsqueeze(1).repeat(1, self.num_samples, 1)
        else:
            if len(agg_encoding.shape) != 3 or agg_encoding.shape[1] != self.num_samples:
                raise Exception('Expected ' + str(self.num_samples) + 'encodings for each train/val data')

        nbr_encodings = inputs['nbr_encodings']

        if nbr_encodings.shape[0] != 32:
            print("nbr_encodings shape: ", nbr_encodings.shape)
            

        # print('nbr_encodings: ', nbr_encodings.shape)

        # Sample latent variable and concatenate with aggregated encoding
        batch_size = agg_encoding.shape[0]
        z = torch.randn(batch_size, self.num_samples, self.lv_dim, device=device)
        agg_encoding = torch.cat((agg_encoding, z), dim=2)
        h = self.leaky_relu(self.hidden(agg_encoding))

        # Your input tensor with shape [32, 161, 32]
        # input_tensor = torch.randn(32, 161, 32)

        # Reshape the input tensor to match the linear layer's input size
        # You need to reshape it to [32, 161 * 32] to match the in_features of the linear layer
        # input_tensor_reshaped = nbr_encodings.view(32, -1)

        # Pass the reshaped tensor through the linear layer
        # output = linear_layer(input_tensor_reshaped)
        h2= self.leaky_relu(self.hidden_2(nbr_encodings))


        # Define the target shape
        target_shape = h.shape

        # # Assuming you have variables x and y
        # x, y = 32, 212  # Change these values as needed

        # # Create a tensor of shape [x, y, 128]
        # input_tensor = torch.randn(x, y, 128)

        # Calculate the padding needed for each dimension
        padding_dims = [max(target - current, 0) for current, target in zip(h2.shape, target_shape)]

        # Pad the input tensor with zeros to the target shape
        h2 = torch.nn.functional.pad(h2, (0, padding_dims[2], 0, padding_dims[1], 0, padding_dims[0]))


        # print(padded_tensor.shape)

        # print('h: ', h.shape)
        # print('h2: ', h2.shape)
        # print('cat: ', torch.cat((h, h2), dim=2).shape)
        # size_diff_0 = h.shape[0] - h2.shape[0]
        # size_diff_1 = h.shape[1] - h2.shape[1]


        # # If tensor2 is smaller, pad it with zeros along the second dimension
        # if size_diff_0 > 0 or size_diff_1 > 0:
        #     padding = torch.zeros((size_diff_0, size_diff_1, 128), device=device)
        #     h2 = torch.cat((h2, padding), dim=0)
        
        
        # print("h2 shape: ", h2.shape)

        # Now you can concatenate the two tensors along the second dimension (dim=1)
        merged_tensor = torch.cat((h, h2), dim=1)
        
        # print(merged_tensor.shape)

        h3 = self.leaky_relu(self.hidden_3(merged_tensor.permute(0, 2, 1)))

        h3 = h3.permute(0, 2, 1)

        # print(h3.shape)
        # print("------------")

        # Output trajectories
        traj = self.op_traj(h3)
        traj = traj.reshape(batch_size, self.num_samples, self.op_len, 2)

        # Cluster
        traj_clustered, probs = cluster_traj(self.num_clusters, traj)

        predictions = {'traj': traj_clustered, 'probs': probs}

        if type(inputs) is dict:
            for key, val in inputs.items():
                if key != 'agg_encoding':
                    predictions[key] = val
        
        # print(predictions['traj'].shape)

        return predictions
