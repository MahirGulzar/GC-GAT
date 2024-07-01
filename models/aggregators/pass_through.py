import torch
import torch.nn as nn
from models.aggregators.aggregator import PredictionAggregator
from typing import Dict
from torch.distributions import Categorical


# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PassThrough(PredictionAggregator):

    def __init__(self):
        super().__init__()

        self.h1 = nn.Linear(32, 128)
        self.h2 = nn.Linear(128, 32)
        self.goal = nn.Linear(5248, 32)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, encodings: Dict) -> Dict:

        # Unpack encodings:
        target_agent_encoding = encodings['target_agent_encoding']
        node_encodings = encodings['context_encoding']['combined']
        node_masks = encodings['context_encoding']['combined_masks']
        intersection_encodings = encodings['context_encoding']['intersection_encoding']
        intersection_masks = encodings['context_encoding']['intersection_masks']
        stopline_encodings = encodings['context_encoding']['stopline_encoding']
        stopline_masks = encodings['context_encoding']['stopline_masks']
        crosswalk_encodings = encodings['context_encoding']['crosswalk_encoding']
        crosswalk_masks = encodings['context_encoding']['crosswalk_masks']

        # print("target_agent_encoding.shape: ", target_agent_encoding.shape)
        # print("node_encodings.shape: ", node_encodings.shape)
        # print("intersection_encodings.shape: ", intersection_encodings.shape)
        # print("stopline_encodings.shape: ", stopline_encodings.shape)
        # print("crosswalk_encodings.shape: ", crosswalk_encodings.shape)


        # agg_encoding = torch.cat((node_encodings, 
        #                           intersection_encodings, 
        #                           stopline_encodings, 
        #                           crosswalk_encodings), dim=1)
        
        agg_encoding = self.leaky_relu(self.h1(node_encodings))
        agg_encoding = self.leaky_relu(self.h2(agg_encoding))

        agg_encoding = torch.reshape(agg_encoding, (agg_encoding.shape[0], -1))

        # agg_encoding = torch.reshape(agg_encoding, (agg_encoding.shape[0], -1))
        # print("agg_encoding.shape: ", agg_encoding.shape)

        agg_encoding = self.leaky_relu(self.goal(agg_encoding))

        # print("agg_encoding.shape: ", agg_encoding.shape)

        agg_encoding = agg_encoding.unsqueeze(1).repeat(1, 1, 1)

        target_agent_encoding = target_agent_encoding.unsqueeze(1).repeat(1, 1, 1)

        agg_encoding = torch.cat((agg_encoding, target_agent_encoding), dim=1)

        

        
        
        # print("agg_encoding.shape: ", agg_encoding.shape)

        

        # Return outputs
        outputs = {'agg_encoding': agg_encoding}

        return outputs