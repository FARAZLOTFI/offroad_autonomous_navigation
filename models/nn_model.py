"""
	The model
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class predictive_model_badgr(nn.Module):
    def __init__(self, planning_horizon, num_of_events):
        # the architecture in brief:
        # three CNN layers with RELU activation fcn
        # + 4 fully connected_layer
        # + LSTM network
        # + two-layer MLP for each action input
        super(predictive_model_badgr, self).__init__()
        self.planning_horizon = planning_horizon
        # CNN layers
        self.CNN_layer1 = nn.Conv2d(3,32,[5,5],2)
        self.CNN_layer2 = nn.Conv2d(32,64,[3,3],2)
        self.CNN_layer3 = nn.Conv2d(64, 64, [3, 3],2)
        self.flatten = nn.Flatten()
        # MLP mayers
        self.MLP_layer1 = nn.Linear(6272,256)
        self.MLP_layer2 = nn.Linear(256, 256)
        self.MLP_layer3 = nn.Linear(256, 128)
        self.MLP_layer4 = nn.Linear(128, 128)

        self.control_action_mlps_layer1 = []
        self.control_action_mlps_layer2 = []
        # last fully connected layer
        self.MLP_last_layers = []

        for i in range(planning_horizon):
            self.control_action_mlps_layer1.append(nn.Linear(1,32))
            self.control_action_mlps_layer2.append(nn.Linear(32,32))
            # last fully connected layer
            self.MLP_last_layers.append(nn.Linear(128, num_of_events))
        # LSTM network takes a set of actions and predicts the corresponding events that might happen
        self.LSTM_network = nn.LSTM(32,128)


    def forward(self, x, actions):
        x = F.relu(self.CNN_layer1(x))
        x = F.relu(self.CNN_layer2(x))
        x = F.relu(self.CNN_layer3(x))
        x = self.flatten(x)

        x = F.relu(self.MLP_layer1(x))
        x = F.relu(self.MLP_layer2(x))
        x = F.relu(self.MLP_layer3(x))
        x = F.relu(self.MLP_layer4(x))
        x = torch.unsqueeze(x,0)
        processed_actions = []
        predicted_output = torch.rand(self.planning_horizon)
        for i,action in zip(range(self.planning_horizon),actions):
            processed_action = self.control_action_mlps_layer1[i](action)
            processed_action = self.control_action_mlps_layer2[i](processed_action)
            processed_actions.append(processed_action)

        processed_actions = torch.stack(processed_actions)

        x = self.LSTM_network(processed_actions,(x,x))[0]

        outputs = []
        for i in range(self.planning_horizon):
            output = self.MLP_last_layers[i](x[i])
            outputs.append(output)
        outputs = torch.stack(outputs)
        # we have to determine for instance some of these must be categories, while some might be regression
        return outputs

if __name__ == '__main__':
    planning_horizon = 10
    num_of_events = 4
    batch_size = 20
    predictive_model = predictive_model_badgr(planning_horizon, num_of_events)
    input_image = torch.rand((batch_size,3,128,72))
    actions = torch.rand(10,batch_size,1)
    # for i in range(planning_horizon):
    #     actions.append(torch.rand(1,1))
    predictive_model(input_image, actions)
