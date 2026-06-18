from typing import Union, List

import torch.nn as nn
import torch
import numpy as np

class BaseNN(nn.Module):
    activations = {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'leaky_relu': nn.LeakyReLU(),
        'elu': nn.ELU(),
        'selu': nn.SELU(),
        'silu': nn.SiLU(),
        'gelu': nn.GELU(),
        'softplus': nn.Softplus(),
        'none': nn.Identity()
    }

    def __init__(self, input_size: int,
                 hidden_size: Union[int, List[int]],
                 output_size: int,
                 dropout_rate: float=0.0,
                 use_batchnorm: bool=False,
                 activation='relu'):
        super(BaseNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm
        self.activation = activation

    def forward(self, x):
        raise NotImplementedError

    def __get_constructor_parameters(self):
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'dropout_rate': self.dropout_rate,
            'use_batchnorm': self.use_batchnorm,
            'activation': self.activation
        }

    def save(self, path):

        torch.save({'state_dict': self.state_dict(), 'data': self.__get_constructor_parameters()}, path)

    @classmethod
    def load(cls, path, device: Union[torch.device, str] = 'auto'):

        if device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)

        checkpoint = torch.load(path, weights_only=False, map_location=device)
        model = cls(**checkpoint["data"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        return model


class FeedForwardNN(BaseNN):

    def __init__(self, input_size, output_size, hidden_size=[], dropout_rate=0.0, use_batchnorm=False, activation='relu'):
        super(FeedForwardNN, self).__init__(input_size, hidden_size, output_size, dropout_rate, use_batchnorm, activation=activation)

        layers = []
        previous_dim = input_size
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]
        for h in hidden_size:
            layers.append(nn.Linear(previous_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(self.activations[activation])
            layers.append(nn.Dropout(p=dropout_rate))
            previous_dim = h
        layers.append(nn.Linear(previous_dim, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class RecurrentNN(BaseNN):

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 dropout_rate=0.0,
                 use_batchnorm=False,
                 activation='relu',
                 bidirectional: bool = False):

        super(RecurrentNN, self).__init__(input_size, hidden_size, output_size, dropout_rate, use_batchnorm, activation=activation)

        assert isinstance(hidden_size, int), "hidden_size must be an integer"
        self.network = nn.GRU(input_size, hidden_size, batch_first=True, dropout=dropout_rate, bidirectional=bidirectional)
        self.head = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
        self.num_directions = 2 if bidirectional else 1

    @property
    def hidden_state_shape(self):
        return self.network.num_layers * self.num_directions, self.hidden_size

    def forward(self, x, h=None):

        if h is None:
            out, last_h = self.network(x)
        else:
            out, last_h = self.network(x, h)

        out = self.head(out)

        return out, last_h


class Curiosity(nn.Module):

    def __init__(self, state_dim, action_logits_dim, action_dim, latent_state_dim, hidden_size):
        super(Curiosity, self).__init__()

        self.name = 'curiosity'

        self.state_feature_extractor = FeedForwardNN(state_dim, hidden_size, latent_state_dim)

        self.inverse_dyn_model = FeedForwardNN(2*latent_state_dim, hidden_size, action_logits_dim)

        self.forward_dyn_model = FeedForwardNN(latent_state_dim + action_dim, hidden_size, latent_state_dim)


    def forward(self, s1, s2, a):

        if isinstance(s1, np.ndarray):
            s1 = torch.tensor(s1, dtype=torch.float32)
        if isinstance(s2, np.ndarray):
            s2 = torch.tensor(s2, dtype=torch.float32)
        if isinstance(a, np.ndarray):
            a = torch.tensor(a, dtype=torch.float32)

        s1_latent = self.state_feature_extractor(s1)
        s2_latent = self.state_feature_extractor(s2)

        a_logits = self.inverse_dyn_model(torch.cat([s1_latent, s2_latent], dim=-1))

        a_logits = a_logits.view(a_logits.size(0), a.size(-1), -1)  # Reshape to (batch_size, max_mol_out, action_dim)

        a_pred = torch.argmax(a_logits, dim=-1).squeeze(-1).float()  # Get the predicted action

        s2_latent_pred = self.forward_dyn_model(torch.cat([s1_latent, a], dim=-1))

        return s2_latent, s2_latent_pred, a_pred
