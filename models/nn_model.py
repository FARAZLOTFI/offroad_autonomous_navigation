"""
	The BADGR baseline model
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class predictive_model_badgr(nn.Module):
    def __init__(self, planning_horizon, num_of_events, action_dimension, shared_mlps = True):
        # the architecture in brief:
        # three CNN layers with RELU activation fcn
        # + 4 fully connected_layer
        # + LSTM network
        # + two-layer MLP for each action input
        super(predictive_model_badgr, self).__init__()
        self.planning_horizon = planning_horizon
        # CNN layers
        self.CNN_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2)
        self.batchNormlization_layer1 = nn.BatchNorm2d(num_features=32)
        self.CNN_layer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.batchNormlization_layer2 = nn.BatchNorm2d(num_features=64)
        self.CNN_layer3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2)
        self.batchNormlization_layer3 = nn.BatchNorm2d(num_features=64)
        self.flatten = nn.Flatten()
        # MLP layers
        self.MLP_layer1 = nn.Linear(6272,256)
        self.MLP_layer2 = nn.Linear(256, 256)
        self.MLP_layer3 = nn.Linear(256, 128)
        self.MLP_layer4 = nn.Linear(128, 128)

        self.control_action_mlps_layer1 = []
        self.control_action_mlps_layer2 = []
        # last fully connected layer
        self.MLP_last_layers1 = []
        self.MLP_last_layers2 = []
        self.MLP_last_layers3 = []

        if shared_mlps:
            linear_layer1 = nn.Linear(action_dimension, 32)
            linear_layer2 = nn.Linear(32, 32)
            output_linear_layer1 = nn.Linear(128, 64)
            output_linear_layer2 = nn.Linear(64, 32)
            output_linear_layer3 = nn.Linear(32, num_of_events)

            for i in range(planning_horizon):
                self.control_action_mlps_layer1.append(linear_layer1)
                self.control_action_mlps_layer2.append(linear_layer2)
                # last fully connected layer
                self.MLP_last_layers1.append(output_linear_layer1)
                self.MLP_last_layers2.append(output_linear_layer2)
                self.MLP_last_layers3.append(output_linear_layer3)
        else:
            for i in range(planning_horizon):
                self.control_action_mlps_layer1.append(nn.Linear(action_dimension,32))
                self.control_action_mlps_layer2.append(nn.Linear(32,32))
                # last fully connected layer
                self.MLP_last_layers1.append(nn.Linear(128, 64))
                self.MLP_last_layers2.append(nn.Linear(64, 32))
                self.MLP_last_layers3.append(nn.Linear(32, num_of_events))

        # make torch lists
        self.control_action_mlps_layer1 = nn.ModuleList(self.control_action_mlps_layer1)
        self.control_action_mlps_layer2 = nn.ModuleList(self.control_action_mlps_layer2)
        self.MLP_last_layers1 = nn.ModuleList(self.MLP_last_layers1)
        self.MLP_last_layers2 = nn.ModuleList(self.MLP_last_layers2)
        self.MLP_last_layers3 = nn.ModuleList(self.MLP_last_layers3)
        # LSTM network takes a set of actions and predicts the corresponding events that might happen
        self.LSTM_network = nn.LSTM(input_size=32, hidden_size=128)


    def extract_features(self,x):
        x = F.relu(self.CNN_layer1(x))
        x = self.batchNormlization_layer1(x)
        x = F.relu(self.CNN_layer2(x))
        x = self.batchNormlization_layer2(x)
        x = F.relu(self.CNN_layer3(x))
        x = self.batchNormlization_layer3(x)
        x = self.flatten(x)

        x = F.relu(self.MLP_layer1(x))
        x = F.relu(self.MLP_layer2(x))
        x = F.relu(self.MLP_layer3(x))
        x = F.relu(self.MLP_layer4(x))
        return x
    def predict_events(self, extracted_features, actions):
        x = torch.unsqueeze(extracted_features, 0)
        processed_actions = []
        predicted_output = torch.rand(self.planning_horizon)
        for i,action in zip(range(self.planning_horizon),actions):
            processed_action = F.relu(self.control_action_mlps_layer1[i](action))
            processed_action = self.control_action_mlps_layer2[i](processed_action)
            processed_actions.append(processed_action)

        processed_actions = torch.stack(processed_actions)

        x, hidden = self.LSTM_network(processed_actions,(x,x))

        outputs = []
        for i in range(self.planning_horizon):
            output = self.MLP_last_layers1[i](x[i])
            output = F.relu(self.MLP_last_layers2[i](output))
            output = self.MLP_last_layers3[i](output)
            outputs.append(output)
        outputs = torch.stack(outputs)
        # we have to determine for instance some of these must be categories, while some might be regression
        return outputs[:,:,:-1], outputs[:,:,-1] # .softmax(dim=-1)

    def training_phase_output(self, input_data):
        input_image, input_actions = input_data
        x = self.extract_features(input_image)
        return self.predict_events(x,input_actions)

if __name__ == '__main__':
    planning_horizon = 10
    num_of_events = 8 + 1  # one for regression
    batch_size = 3
    action_dimension = 2
    predictive_model = predictive_model_badgr(planning_horizon, num_of_events, action_dimension)
    input_image = torch.rand((batch_size,3,128,72))
    actions = torch.rand(planning_horizon,batch_size,action_dimension)
    # we'll have two steps
    # the first step is to extract features of the current observation which is run only once!
    # the second step,however, will be iterative to find the best set of actions given the current latent variable
    x = predictive_model.extract_features(input_image)
    out = predictive_model.predict_events(x,actions)
