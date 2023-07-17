"""
	The BADGR baseline model
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class predictive_model_badgr(nn.Module):
    def __init__(self, planning_horizon, num_event_types, action_dimension, 
                 n_seq_model_layers, shared_mlps=True):
        # the architecture in brief:
        # three CNN layers with RELU activation fcn
        # + 4 fully connected_layer
        # + LSTM network
        # + two-layer MLP for each action input
        super(predictive_model_badgr, self).__init__()
        self.planning_horizon = planning_horizon

        self.n_seq_model_layers = n_seq_model_layers
        self.hidden_size = 16

        # input embedder
        self.input_embedder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(6272, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * self.hidden_size * n_seq_model_layers),
            nn.ReLU()
        )

        self.action_embedder = nn.ModuleList()
        self.event_head = nn.ModuleList()

        if shared_mlps:
            action_embed_seq = nn.Sequential(
                nn.Linear(action_dimension, 16), 
                nn.ReLU(),
                nn.Linear(16, 16)
            )
            event_seq = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, num_event_types)
            )

            for _ in range(planning_horizon):
                self.action_embedder.append(action_embed_seq)
                self.event_head.append(event_seq)

        else:
            for _ in range(planning_horizon):
                action_embed_seq = nn.Sequential(
                    nn.Linear(action_dimension, 16), 
                    nn.ReLU(), 
                    nn.Linear(16, 16),
                )
                self.action_embedder.append(action_embed_seq)
                event_seq = nn.Sequential(
                    nn.Linear(16, 32),
                    nn.ReLU(),
                    nn.Linear(32, num_event_types)
                )
                self.event_head.append(event_seq)

        # LSTM network takes a set of actions and predicts the corresponding 
         # events that might happen
        self.LSTM_network = nn.LSTM(input_size=16, 
                                    hidden_size=self.hidden_size, 
                                    num_layers=n_seq_model_layers)

    def extract_features(self,x):
        x = self.input_embedder(x)
        return x
    
    def predict_events(self, extracted_features, actions):
        x = torch.unsqueeze(extracted_features, 0)
        processed_actions = []
        for i,action in zip(range(self.planning_horizon), actions):
            processed_action = self.action_embedder[i](action)
            processed_actions.append(processed_action)

        processed_actions = torch.stack(processed_actions)

        x = x.reshape(self.n_seq_model_layers, -1, 
                      2 * processed_actions.shape[-1])
        h_0, c_0 = torch.chunk(x, 2, dim=-1)
        x, _ = self.LSTM_network(processed_actions, (h_0, c_0))

        outputs = []
        for i in range(self.planning_horizon):
            output = self.event_head[i](x[i])
            outputs.append(output)
        outputs = torch.stack(outputs)
        # we have to determine for instance some of these must be categories, 
         # while some might be regression
        return outputs[:,:,:-1], outputs[:,:,-1] # .softmax(dim=-1)

    def training_phase_output(self, input_data):
        input_image, input_actions = input_data
        x = self.extract_features(input_image)
        return self.predict_events(x,input_actions)

if __name__ == '__main__':
    planning_horizon = 10
    num_event_types = 8 + 1  # one for regression
    batch_size = 3
    action_dimension = 2
    n_seq_model_layers = 4
    predictive_model = predictive_model_badgr(planning_horizon, num_event_types, 
                                              action_dimension, 
                                              n_seq_model_layers)
    input_image = torch.rand((batch_size,3,128,72))
    actions = torch.rand(planning_horizon,batch_size,action_dimension)
    # we'll have two steps
     # the first step is to extract features of the current observation which 
     # is run only once! the second step,however, will be iterative to find the 
     # best set of actions given the current latent variable.
    x = predictive_model.extract_features(input_image)
    out = predictive_model.predict_events(x,actions)
