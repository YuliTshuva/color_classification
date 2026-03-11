"""
Yuli Tshuva
Implementing the models for the GNN.

# Code usage:
GNN(in_features=COLOR_EMBEDDING_DIM, hidden_dim=GNN_HIDDEN_DIM,
    out_features=GNN_EMBEDDING_DIM, K=K_GNN_LAYERS)
MLP(input_dim=GNN_EMBEDDING_DIM, hidden_dims=MLP_HIDDEN_DIMS, output_dim=n_colors)
"""

# Imports
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from constants import *


class GNN(torch.nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, K, dropout_rate: float):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.k = K

        if K >= 3:
            self.conv_layers = nn.ModuleList([GCNConv(in_features, hidden_dim)] +
                                             [GCNConv(hidden_dim, hidden_dim) for _ in range(K - 2)] +
                                             [GCNConv(hidden_dim, out_features)])
        elif K == 2:
            self.conv_layers = nn.ModuleList([GCNConv(in_features, hidden_dim), GCNConv(hidden_dim, out_features)])
        elif K == 1:
            self.conv_layers = nn.ModuleList([GCNConv(in_features, out_features)])
        else:
            raise NotImplementedError

    def forward(self, x, edge_index):
        # Hidden layers
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)

        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate: float):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class ColorEmbedding(nn.Module):
    def __init__(self, n_colors, embedding_dim):
        """
        Initialize the model.
        :param n_colors: Amount of colors our graph contains.
        """
        super(ColorEmbedding, self).__init__()
        # Define an embedding layer for the colors
        self.color_embedding = nn.Embedding(n_colors, embedding_dim)

    def forward(self, x):
        """
        Forward pass of the model.
        :param x: The input color indices. Shape [num_nodes].
        :return: The output of the model. Shape [num_nodes, COLOR_EMBEDDING_DIM].
        """
        return self.color_embedding(x)


class AttentionColorClassifier(nn.Module):
    def __init__(self, dim, num_heads, num_classes):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.mlp = MLP(input_dim=dim, hidden_dims=ATTENTION_HIDDEN_DIMS, output_dim=num_classes,
                       dropout_rate=ATTENTION_DROPOUT_RATE)

    def forward(self, x):
        # Self-Attention
        attn_output, _ = self.attn(x, x, x)
        # Add & Norm
        x = self.norm(attn_output + x)
        # Classification
        return self.mlp(x)
