# backend/app/services/gnn_model.py

import torch
from torch import nn
from torch_geometric.nn import SAGEConv


class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.acts.append(nn.ReLU())

        # Hidden layers (if >2)
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.acts.append(nn.ReLU())

        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.acts.append(nn.Identity())  # no activation on last layer

    def forward(self, x, edge_index):
        for conv, act in zip(self.convs, self.acts):
            x = conv(x, edge_index)
            x = act(x)
            x = self.dropout(x)
        return x
