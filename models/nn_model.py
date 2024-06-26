import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class PredictiveModelBadgr(nn.Module):
    def __init__(self, planning_horizon, num_event_types, action_dimension, 
                 seq_model, n_seq_model_layers = 2, shared_mlps=True, 
                 ensemble_size=1, device='cuda', ensemble_type='fixed_masks'):
        # the architecture in brief:
        # three CNN layers with RELU activation fcn
        # + 4 fully connected_layer
        # + LSTM network
        # + two-layer MLP for each action input
        super(PredictiveModelBadgr, self).__init__()
        self.planning_horizon = planning_horizon
        if isinstance(seq_model, nn.ModuleList):
            input_dim = seq_model[0].input_dim
        else:
            input_dim = seq_model.input_dim

        self.n_seq_model_layers = n_seq_model_layers

        if type(seq_model) is LSTMSeqModel:
            # make the state embedder output all the hidden vectors of the LSTM
            state_embed_dim = \
                2 * seq_model.hidden_dim * seq_model.n_seq_model_layers
        else:
            # the state embedder just outputs a single vector
            state_embed_dim = input_dim

        self.state_embedder = nn.Sequential(
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
            nn.Linear(128, state_embed_dim),
            nn.ReLU()
        )
        self.action_embedder = nn.ModuleList()
        self.event_head = nn.ModuleList()

        self.ensemble_size = ensemble_size
        self.ensemble_type = ensemble_type
        # define a function to instantiate MLPs
        def build_mlps():
            action_embed_seq = nn.Sequential(
                nn.Linear(action_dimension, input_dim), 
                nn.ReLU(),
                nn.Linear(input_dim, input_dim)
            )
            if ensemble_size == 1:
                event_seq = nn.Sequential(
                    nn.Linear(input_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, num_event_types)
                )
            else:
                if ensemble_type == 'fixed_masks':
                    shared_layer = nn.Sequential(
                        nn.Linear(input_dim, 32*ensemble_size),
                        nn.ReLU()
                        )
                    self.create_masks(ensemble_size, device)
                    heads = nn.Linear(32*ensemble_size, num_event_types+1)
                elif ensemble_type == 'multihead':
                    shared_layer = nn.Sequential(
                        nn.Linear(input_dim, 32),
                        nn.ReLU()
                        )
                    heads = nn.ModuleList([
                        nn.Sequential(
                            nn.Linear(32, num_event_types+1)
                        ) for i in range(ensemble_size)
                    ])
                event_seq = nn.ModuleList([shared_layer, heads])    
            return action_embed_seq, event_seq

        if shared_mlps:
            # instantiate the MLPs once and share them across the planning 
            action_embed_seq, event_seq = build_mlps()
        for _ in range(planning_horizon):
            if not shared_mlps:
                # instantiate separate MLPs for each planning step
                action_embed_seq, event_seq = build_mlps()
            self.action_embedder.append(action_embed_seq)
            self.event_head.append(event_seq)

        # LSTM network takes a set of actions and predicts the corresponding 
         # events that might happen
        self.seq_encoder = seq_model

    def create_masks(self, num_masks, device):
        masks = []
        for i in range(num_masks):
            mask_l1 = torch.bernoulli(torch.full_like(torch.ones(32*self.ensemble_size), 0.5))\
                .to(device)
            masks.append(mask_l1)
        self.masks = masks

    def extract_features(self,x):
        x = self.state_embedder(x)
        return x
    
    def predict_events(self, state_embed, actions, ensemble_comp = -1):
        ## NOTE if ensemble_comp < 0 picks a random component, 
        ## random components should be used during training, 
        ## though when estimating uncertainty at test time
        ## one should iterate over the components
        state_embed = torch.unsqueeze(state_embed, 0)
        processed_actions = []
        for i,action in zip(range(self.planning_horizon), actions):
            processed_action = self.action_embedder[i](action)
            processed_actions.append(processed_action)

        processed_actions = torch.stack(processed_actions)
        if ensemble_comp < 0:
            ensemble_comp = np.random.choice(self.ensemble_size)
        if isinstance(self.seq_encoder, nn.ModuleList):
            seq_embeddings = self.seq_encoder[ensemble_comp](state_embed, processed_actions)
        else:
            seq_embeddings = self.seq_encoder(state_embed, processed_actions)

        # this could be sped up by packing the embeddings into a batch and
         # passing them through the network all at once.
        outputs = []
        for i in range(self.planning_horizon):
            if self.ensemble_size > 1:
                if ensemble_comp < 0:
                    ensemble_comp = np.random.choice(self.ensemble_size)
                if self.ensemble_type =='fixed_masks':
                    output = self.event_head[i][0](seq_embeddings[i])*self.masks[ensemble_comp]
                    output = self.event_head[i][1](output)
                elif self.ensemble_type =='multihead':
                    output = self.event_head[i][0](seq_embeddings[i])
                    output = self.event_head[i][1][ensemble_comp](output)
            else:
                output = self.event_head[i](seq_embeddings[i])
            outputs.append(output)
        outputs = torch.stack(outputs)
        # we have to determine for instance some of these must be categories, 
         # while some might be regression
        if self.ensemble_size > 1:
            return outputs[:,:,:-2], outputs[:,:,-2:] # .softmax(dim=-1)
        return outputs[:,:,:-1], outputs[:,:,-1] # .softmax(dim=-1)

    def training_phase_output(self, input_data, ensemble_comp = -1):
        input_image, input_actions = input_data
        x = self.extract_features(input_image)
        return self.predict_events(x, input_actions, ensemble_comp = ensemble_comp)



class SeqModel(nn.Module):
    def __init__(self, n_seq_model_layers, input_dim=16):
        super().__init__()
        self.n_seq_model_layers = n_seq_model_layers
        self.input_dim = input_dim

    @property
    def output_dim(self):
        # by default, output dim is the same as input dim.
        return self.input_dim


class LSTMSeqModel(SeqModel):
    def __init__(self, n_seq_model_layers, input_dim=16, hidden_dim=16):
        super().__init__(n_seq_model_layers, input_dim)
        self.hidden_dim = hidden_dim
        self.LSTM_network = nn.LSTM(input_size=input_dim, 
                                    hidden_size=hidden_dim, 
                                    num_layers=n_seq_model_layers)

    def forward(self, state_embed, action_embeddings):
        ext_feats = state_embed.reshape(
            self.n_seq_model_layers, -1, 
            2 * action_embeddings.shape[-1])
        h_0, c_0 = torch.chunk(ext_feats, 2, dim=-1)
        seq_embed, _ = self.LSTM_network(action_embeddings, (h_0.contiguous(), c_0.contiguous()))

        return seq_embed
    

class CustomEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super(CustomEncoderLayer, self).__init__()

        # Multi-Head Self-Attention Layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead)

        # Feedforward Neural Network Layer
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, x, src_mask=None, is_causal=None, src_key_padding_mask=None):
        # Multi-Head Self-Attention
        x = self.self_attn(x, x, x, attn_mask=src_mask)[0]

        # Feedforward Neural Network
        x = self.feedforward(x)

        return x





class TransformerSeqModel(SeqModel):
    def __init__(self, n_seq_model_layers, input_dim=16, n_heads=4):
        super().__init__(n_seq_model_layers, input_dim)
        tf_layer = nn.TransformerEncoderLayer(d_model=input_dim,
                                              nhead=n_heads,
                                              dim_feedforward=input_dim*4)
        self.tf_network = nn.TransformerEncoder(tf_layer, 
                                                num_layers=n_seq_model_layers)
                                                #norm=nn.LayerNorm(input_dim))

    def forward(self, state_embed, action_embeddings):
        # add a sequence embedding
        seqlen = action_embeddings.shape[0]
        sinseq = get_sinusoid_pos_embeddings(seqlen, state_embed.shape[-1])
        sinseq = sinseq.to(action_embeddings.device)
        sinseq = sinseq.unsqueeze(1).expand(-1, action_embeddings.shape[1], -1)
        action_embeddings = action_embeddings + sinseq

        # concatenate the state embedding with the action embeddings
         # we do this *after* the positional encoding so the state embedding
         # doesn't get one
        sequence = torch.cat((state_embed, action_embeddings), dim=0)
        seq_embed = self.tf_network(sequence)
        # remove the embedding of the state
        seq_embed = seq_embed[1:]

        return seq_embed


def get_sinusoid_pos_embeddings(seqlen, ndims, posenc_min_rate=1/10000):
    angle_rate_exps = torch.linspace(0, 1, ndims // 2)
    angle_rates = posenc_min_rate ** angle_rate_exps
    positions = torch.arange(seqlen)
    angles_rad = positions[:, None] * angle_rates[None, :]
    sines = torch.sin(angles_rad)
    cosines = torch.cos(angles_rad)
    return torch.cat((sines, cosines), dim=-1)


if __name__ == '__main__':
    planning_horizon = 10
    num_event_types = 8 + 1  # one for regression
    batch_size = 3
    action_dimension = 2
    n_seq_model_layers = 4
    seq_elem_dim = 16
    seq_encoder = LSTMSeqModel(n_seq_model_layers, seq_elem_dim)
    predictive_model = PredictiveModelBadgr(planning_horizon, num_event_types, 
                                            action_dimension, seq_encoder)
    input_image = torch.rand((batch_size, 3, 128, 72))
    actions = torch.rand(planning_horizon, batch_size, action_dimension)
    # we'll have two steps
     # the first step is to extract features of the current observation which 
     # is run only once! the second step,however, will be iterative to find the 
     # best set of actions given the current latent variable.
    x = predictive_model.extract_features(input_image)
    out = predictive_model.predict_events(x,actions)

    # now test the transformer
    seq_encoder = TransformerSeqModel(n_seq_model_layers, seq_elem_dim)
    predictive_model = PredictiveModelBadgr(planning_horizon, num_event_types,
                                            action_dimension, seq_encoder)
    x = predictive_model.extract_features(input_image)
    out = predictive_model.predict_events(x,actions)
