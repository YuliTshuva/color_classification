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
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from constants import *


class GCN(torch.nn.Module):
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


class GAT(torch.nn.Module):
    """
    Graph Attention Network built on PyG's GATConv.

    Intermediate layers use multi-head attention with concatenation (so their
    output dim is hidden_dim * heads). The final layer uses a single head with
    averaging (concat=False) to produce out_features logits.
    """

    def __init__(self, in_features, hidden_dim, out_features, K, dropout_rate: float, heads: int = 1):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.k = K
        self.heads = heads

        if K >= 3:
            self.conv_layers = nn.ModuleList(
                [GATConv(in_features, hidden_dim, heads=heads, concat=True, dropout=dropout_rate)] +
                [GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True, dropout=dropout_rate)
                 for _ in range(K - 2)] +
                [GATConv(hidden_dim * heads, out_features, heads=1, concat=False, dropout=dropout_rate)])
        elif K == 2:
            self.conv_layers = nn.ModuleList(
                [GATConv(in_features, hidden_dim, heads=heads, concat=True, dropout=dropout_rate),
                 GATConv(hidden_dim * heads, out_features, heads=1, concat=False, dropout=dropout_rate)])
        elif K == 1:
            self.conv_layers = nn.ModuleList(
                [GATConv(in_features, out_features, heads=1, concat=False, dropout=dropout_rate)])
        else:
            raise NotImplementedError

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            if i < len(self.conv_layers) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)

        return x


class GraphSAGE(torch.nn.Module):
    """
    GraphSAGE built on PyG's SAGEConv. Same K-layer depth logic as GCN.
    """

    def __init__(self, in_features, hidden_dim, out_features, K, dropout_rate: float):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.k = K

        if K >= 3:
            self.conv_layers = nn.ModuleList([SAGEConv(in_features, hidden_dim)] +
                                             [SAGEConv(hidden_dim, hidden_dim) for _ in range(K - 2)] +
                                             [SAGEConv(hidden_dim, out_features)])
        elif K == 2:
            self.conv_layers = nn.ModuleList([SAGEConv(in_features, hidden_dim), SAGEConv(hidden_dim, out_features)])
        elif K == 1:
            self.conv_layers = nn.ModuleList([SAGEConv(in_features, out_features)])
        else:
            raise NotImplementedError

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)

        return x


class GIN(torch.nn.Module):
    """
    Graph Isomorphism Network built on PyG's GINConv.

    Each GINConv wraps a small MLP (Linear -> ReLU -> Linear) used as the
    injective aggregation function. Same K-layer depth logic as GCN.
    """

    @staticmethod
    def _mlp(in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def __init__(self, in_features, hidden_dim, out_features, K, dropout_rate: float):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.k = K

        if K >= 3:
            self.conv_layers = nn.ModuleList([GINConv(self._mlp(in_features, hidden_dim))] +
                                             [GINConv(self._mlp(hidden_dim, hidden_dim)) for _ in range(K - 2)] +
                                             [GINConv(self._mlp(hidden_dim, out_features))])
        elif K == 2:
            self.conv_layers = nn.ModuleList([GINConv(self._mlp(in_features, hidden_dim)),
                                              GINConv(self._mlp(hidden_dim, out_features))])
        elif K == 1:
            self.conv_layers = nn.ModuleList([GINConv(self._mlp(in_features, out_features))])
        else:
            raise NotImplementedError

    def forward(self, x, edge_index):
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


class RGCNLayer(nn.Module):
    """
    Relational Graph Convolutional Layer.

    For each relation type r, applies a separate linear transform W_r to
    neighboring features, aggregates (sum + normalize), then adds a self-loop
    transform. All contributions are summed and passed through an activation.
    """

    def __init__(self, in_dim: int, out_dim: int, num_relations: int,
                 self_loop: bool = True, directed: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.self_loop = self_loop
        self.directed = directed

        # One weight matrix per relation: shape (num_relations, in_dim, out_dim)
        self.weight = nn.Parameter(torch.Tensor(num_relations, in_dim, out_dim))

        # Optional separate transform for self-loop
        if self_loop:
            self.self_weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        else:
            self.register_parameter("self_weight", None)

        self.reset_parameters()

    def reset_parameters(self):
        for r in range(self.num_relations):
            nn.init.xavier_uniform_(self.weight[r])
        if self.self_weight is not None:
            nn.init.xavier_uniform_(self.self_weight)

    def forward(
            self,
            x: torch.Tensor,  # [N, in_dim]  node features
            edge_index: torch.Tensor,  # [2, E]       source/target node indices
            edge_type: torch.Tensor,  # [E]          relation id for each edge
    ) -> torch.Tensor:
        N = x.size(0)
        out = torch.zeros(N, self.out_dim, device=x.device)

        # --- Relational aggregation ---
        for r in range(self.num_relations):
            # Mask edges that belong to relation r
            mask = edge_type == r
            if not mask.any():
                continue

            # Define source and target nodes for the current relation
            src = edge_index[0, mask]
            dst = edge_index[1, mask]

            if not self.directed:
                src = torch.cat([src, edge_index[1, mask]])
                dst = torch.cat([dst, edge_index[0, mask]])

            # Single pass over the (now doubled) edge list
            msg = x[src] @ self.weight[r]

            deg = torch.zeros(N, device=x.device)
            deg.scatter_add_(0, dst, torch.ones(src.shape[0], device=x.device))
            deg_inv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))

            agg = torch.zeros(N, self.out_dim, device=x.device)
            agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)

            out += agg * deg_inv.unsqueeze(1)

        # --- Self-loop ---
        if self.self_loop:
            out += x @ self.self_weight

        return out


class RGCN(nn.Module):
    """
    Multi-layer RGCN for node classification.
    """

    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            out_dim: int,
            num_relations: int,
            num_layers: int = 2,
            dropout: float = 0.0,
            directed: bool = False
    ):
        super().__init__()
        self.dropout = dropout

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([
            RGCNLayer(dims[i], dims[i + 1], num_relations, directed=directed)
            for i in range(num_layers)
        ])

    def forward(self, x, edge_index, edge_type):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_type)
            if i < len(self.layers) - 1:  # activation on all but last
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # return logits; apply softmax/sigmoid outside


class RGATLayer(nn.Module):
    """
    Relational Graph Attention Layer.

    For each relation type r, computes a learned attention coefficient
    alpha_{ij}^r between each source-target pair, then uses those
    coefficients (instead of uniform degree-normalization) to aggregate
    neighbor features. A separate self-loop transform is added, and all
    contributions are summed before activation.

    Attention score for edge (i -> j) under relation r:
        e_{ij}^r = LeakyReLU( a_r^T [W_r h_i || W_r h_j] )
        alpha_{ij}^r = softmax over all neighbors of j under r
    """

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            num_relations: int,
            num_heads: int = 1,
            dropout: float = 0.0,
            negative_slope: float = 0.2,
            self_loop: bool = True,
            directed: bool = False,
    ):
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.self_loop = self_loop
        self.directed = directed

        # Per-relation weight matrices: (num_relations, in_dim, out_dim)
        self.weight = nn.Parameter(torch.empty(num_relations, in_dim, out_dim))

        # Per-relation attention vectors: (num_relations, num_heads, 2 * head_dim)
        # The factor of 2 is for the concatenation [W h_i || W h_j]
        self.attn = nn.Parameter(torch.empty(num_relations, num_heads, 2 * self.head_dim))

        if self_loop:
            self.self_weight = nn.Parameter(torch.empty(in_dim, out_dim))
        else:
            self.register_parameter("self_weight", None)

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def reset_parameters(self):
        for r in range(self.num_relations):
            nn.init.xavier_uniform_(self.weight[r])
        nn.init.xavier_uniform_(self.attn.view(self.num_relations, -1).unsqueeze(0))
        if self.self_weight is not None:
            nn.init.xavier_uniform_(self.self_weight)

    def forward(
            self,
            x: torch.Tensor,  # [N, in_dim]
            edge_index: torch.Tensor,  # [2, E]
            edge_type: torch.Tensor,  # [E]
    ) -> torch.Tensor:
        N = x.size(0)
        out = torch.zeros(N, self.out_dim, device=x.device)

        for r in range(self.num_relations):
            mask = edge_type == r
            if not mask.any():
                continue

            src = edge_index[0, mask]
            dst = edge_index[1, mask]

            if not self.directed:
                src = torch.cat([src, edge_index[1, mask]])
                dst = torch.cat([dst, edge_index[0, mask]])

            E_r = src.size(0)

            # Linear transform: [E_r, out_dim] -> [E_r, num_heads, head_dim]
            h_src = (x[src] @ self.weight[r]).view(E_r, self.num_heads, self.head_dim)
            h_dst = (x[dst] @ self.weight[r]).view(E_r, self.num_heads, self.head_dim)

            # Attention logits: concat source and target projections, dot with a_r
            # attn[r]: [num_heads, 2*head_dim]
            # cat([h_src, h_dst], dim=-1): [E_r, num_heads, 2*head_dim]
            e = (torch.cat([h_src, h_dst], dim=-1) * self.attn[r]).sum(dim=-1)
            # e: [E_r, num_heads]
            e = self.leaky_relu(e)

            # Softmax over neighbors of each dst node, per head
            # We use scatter to compute the softmax denominator
            e_max = torch.zeros(N, self.num_heads, device=x.device)
            e_max.scatter_reduce_(0, dst.unsqueeze(1).expand_as(e), e, reduce="amax", include_self=True)
            e_exp = torch.exp(e - e_max[dst])

            denom = torch.zeros(N, self.num_heads, device=x.device)
            denom.scatter_add_(0, dst.unsqueeze(1).expand_as(e_exp), e_exp)
            denom = denom.clamp(min=1e-9)

            alpha = e_exp / denom[dst]  # [E_r, num_heads]

            # Optional attention dropout
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

            # Weighted aggregation: alpha * h_src, then sum at dst
            # [E_r, num_heads, head_dim]
            weighted = alpha.unsqueeze(-1) * h_src

            agg = torch.zeros(N, self.num_heads, self.head_dim, device=x.device)
            agg.scatter_add_(
                0,
                dst.view(-1, 1, 1).expand_as(weighted),
                weighted,
            )

            # Flatten heads back to out_dim and accumulate
            out += agg.view(N, self.out_dim)

        if self.self_loop:
            out += x @ self.self_weight

        return out


class RGAT(nn.Module):
    """
    Multi-layer RGAT for node classification.
    """

    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            out_dim: int,
            num_relations: int,
            num_layers: int = 2,
            num_heads: int = 4,
            dropout: float = 0.0,
            attn_dropout: float = 0.0,
            negative_slope: float = 0.2,
            directed: bool = False,
    ):
        super().__init__()
        self.dropout = dropout

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([
            RGATLayer(
                dims[i], dims[i + 1],
                num_relations=num_relations,
                num_heads=num_heads if i < num_layers - 1 else 1,
                dropout=attn_dropout,
                negative_slope=negative_slope,
                directed=directed,
            )
            for i in range(num_layers)
        ])

    def forward(self, x, edge_index, edge_type):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_type)
            if i < len(self.layers) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # logits; apply softmax/sigmoid outside
