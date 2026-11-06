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


def scatter_softmax(scores, index, n_segments):
    """
    Softmax of `scores` within each segment given by `index`.

    :param scores: [M, H] raw scores (H = heads, use H=1 for a single score).
    :param index: [M] segment id of each row (here: the color of each node).
    :param n_segments: number of segments (n_colors).
    :return: [M, H] weights that sum to 1 within every segment.
    """
    seg_max = torch.full((n_segments, scores.size(1)), float("-inf"), device=scores.device)
    seg_max.scatter_reduce_(0, index.unsqueeze(1).expand_as(scores), scores, reduce="amax", include_self=True)
    # Segments with no members keep -inf; they are never read back through index.
    exp = torch.exp(scores - seg_max[index])

    denom = torch.zeros(n_segments, scores.size(1), device=scores.device)
    denom.scatter_add_(0, index.unsqueeze(1).expand_as(exp), exp)
    return exp / denom[index].clamp(min=1e-9)


def scatter_mean(src, index, n_segments):
    """Mean of `src` rows within each segment. :return: [n_segments, dim]"""
    out = torch.zeros(n_segments, src.size(1), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    counts = torch.bincount(index, minlength=n_segments).clamp(min=1)
    return out / counts.unsqueeze(1).to(src.dtype)


class MeanReadout(nn.Module):
    """
    The exp08 color head: parameter-free mean-pooling of the member node
    embeddings. Kept as a module so every readout shares one interface and the
    comparison is a single-argument swap.
    """

    def __init__(self, dim):
        super().__init__()
        self.out_dim = dim

    def forward(self, h, color_indices, n_colors):
        return scatter_mean(h, color_indices, n_colors)


class MultiStatReadout(nn.Module):
    """
    Proposal 1.3 - concatenate mean || max || sum before the color MLP.

    The mean washes out "one abnormal node" patterns that max preserves, and the
    sum retains the bag size that both mean and max discard. Parameter-free; the
    only cost is a color MLP with a 3x wider input.
    """

    def __init__(self, dim):
        super().__init__()
        self.out_dim = 3 * dim

    def forward(self, h, color_indices, n_colors):
        idx = color_indices.unsqueeze(1).expand_as(h)

        summed = torch.zeros(n_colors, h.size(1), device=h.device, dtype=h.dtype)
        summed.index_add_(0, color_indices, h)

        counts = torch.bincount(color_indices, minlength=n_colors)
        mean = summed / counts.clamp(min=1).unsqueeze(1).to(h.dtype)

        maxed = torch.full((n_colors, h.size(1)), float("-inf"), device=h.device, dtype=h.dtype)
        maxed.scatter_reduce_(0, idx, h, reduce="amax", include_self=True)
        maxed = maxed.masked_fill((counts == 0).unsqueeze(1), 0.0)

        return torch.cat([mean, maxed, summed], dim=1)


class GatedAttentionReadout(nn.Module):
    """
    Proposal 1.1 - gated attention (MIL) pooling, Ilse et al. (2018).

    A small gated MLP scores every node, the scores are softmax-normalised
    *within each color*, and the color representation is the weighted sum
    sum_i alpha_i h_i. Nodes carrying the signal are amplified, uninformative
    ones suppressed. The alphas are returned so they can be inspected: they say
    which visits drove each patient-level prediction.

        a_i = tanh(V h_i) * sigmoid(U h_i)
        e_i = w^T a_i
        alpha_i = softmax_over_color(e_i)
    """

    def __init__(self, dim, hidden_dim, dropout_rate: float = 0.0):
        super().__init__()
        self.out_dim = dim
        self.V = nn.Linear(dim, hidden_dim)
        self.U = nn.Linear(dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.last_alpha = None

    def forward(self, h, color_indices, n_colors):
        a = torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))
        e = self.w(self.dropout(a))  # [N, 1]
        alpha = scatter_softmax(e, color_indices, n_colors)  # [N, 1]
        self.last_alpha = alpha.detach().squeeze(1)

        pooled = torch.zeros(n_colors, h.size(1), device=h.device, dtype=h.dtype)
        pooled.index_add_(0, color_indices, alpha * h)
        return pooled


class EdgeSelfAttention(nn.Module):
    """
    A Set-Transformer SAB (Vaswani-style block: multi-head self-attention ->
    add & norm -> feed-forward -> add & norm) restricted to the pairs given by
    `pair_edge_index`.

    Passing every within-color pair (plus self-loops) makes the attention exactly
    block-diagonal over the bags, i.e. this *is* a standard SAB run independently
    on each color's node set - but computed on the edge list instead of a padded
    [n_colors, max_bag, dim] tensor. That matters here: max_bag is 221 while the
    mean is 9.3, so padding would waste ~95% of the memory.
    """

    def __init__(self, dim, num_heads, dropout_rate: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, h, pair_edge_index):
        N, dim = h.shape
        src, dst = pair_edge_index[0], pair_edge_index[1]

        q = self.q(h).view(N, self.num_heads, self.head_dim)
        k = self.k(h).view(N, self.num_heads, self.head_dim)
        v = self.v(h).view(N, self.num_heads, self.head_dim)

        # Scaled dot-product logits for the kept pairs only: [E, num_heads]
        logits = (q[dst] * k[src]).sum(-1) / (self.head_dim ** 0.5)
        alpha = self.dropout(scatter_softmax(logits, dst, N))

        agg = torch.zeros(N, self.num_heads, self.head_dim, device=h.device, dtype=h.dtype)
        weighted = alpha.unsqueeze(-1) * v[src]
        agg.scatter_add_(0, dst.view(-1, 1, 1).expand_as(weighted), weighted)

        h = self.norm1(h + self.o(agg.view(N, dim)))
        return self.norm2(h + self.ff(h))


class PMAReadout(nn.Module):
    """
    Proposal 1.4 - Pooling by Multihead Attention (Lee et al., 2019).

    A learned seed vector queries the color's node set with multi-head
    attention, optionally after `num_sabs` self-attention blocks among the
    member nodes (the part that makes this strictly more expressive than 1.1:
    it can model interactions *between* members at readout time).

    Set num_sabs=0 for plain PMA; then no within-bag pair list is needed and the
    cost is the same as gated attention, with a learned query instead of a
    learned scorer.
    """

    def __init__(self, dim, num_heads, num_sabs: int = 0, dropout_rate: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.out_dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.sabs = nn.ModuleList([EdgeSelfAttention(dim, num_heads, dropout_rate) for _ in range(num_sabs)])
        self.rff = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())
        self.seed = nn.Parameter(torch.empty(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.seed.unsqueeze(0))

        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.dropout = nn.Dropout(dropout_rate)
        self.last_alpha = None

    def forward(self, h, color_indices, n_colors, pair_edge_index=None):
        for sab in self.sabs:
            if pair_edge_index is None:
                raise ValueError("PMAReadout with num_sabs > 0 needs pair_edge_index")
            h = sab(h, pair_edge_index)

        h = self.rff(h)
        N = h.size(0)
        k = self.k(h).view(N, self.num_heads, self.head_dim)
        v = self.v(h).view(N, self.num_heads, self.head_dim)

        # The query is the shared seed, so the logit of node i is seed . k_i
        logits = (k * self.seed).sum(-1) / (self.head_dim ** 0.5)  # [N, num_heads]
        alpha = self.dropout(scatter_softmax(logits, color_indices, n_colors))
        self.last_alpha = alpha.detach().mean(dim=1)

        agg = torch.zeros(n_colors, self.num_heads, self.head_dim, device=h.device, dtype=h.dtype)
        weighted = alpha.unsqueeze(-1) * v
        agg.scatter_add_(0, color_indices.view(-1, 1, 1).expand_as(weighted), weighted)

        pooled = self.norm(self.seed.reshape(1, -1) + self.o(agg.view(n_colors, -1)))
        return pooled + self.ff(pooled)


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


class AttentionClassifier(nn.Module):
    def __init__(self, dim, num_heads, num_classes, attention_hidden_dims, attention_dropout_rate, n_layers):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.mlp = MLP(input_dim=dim, hidden_dims=attention_hidden_dims, output_dim=num_classes,
                       dropout_rate=attention_dropout_rate)
        self.n_layers = n_layers

    def forward(self, x):
        # Apply multi-head attention and layer normalization for n_layers
        for _ in range(self.n_layers):
            # Self-Attention
            attn_output, _ = self.attn(x, x, x)
            # Add & Norm
            x = self.norm(attn_output + x)
        # Classification
        return self.mlp(x)


class GraphNorm(nn.Module):
    """
    GraphNorm (Cai et al. 2021) over one graph, as GLASS uses it: subtract a
    learned fraction of the mean over the nodes, divide by the std, then scale
    and shift. Same as torch_geometric's GraphNorm with batch=None.
    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.mean_scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        out = x - self.mean_scale * x.mean(dim=0, keepdim=True)
        std = (out.pow(2).mean(dim=0, keepdim=True) + self.eps).sqrt()
        return self.weight * out / std + self.bias


class RGCNLayer(nn.Module):
    """
    Relational Graph Convolutional Layer.

    For each relation type r, applies a separate linear transform W_r to
    neighboring features, aggregates (sum + normalize), then adds a self-loop
    transform. All contributions are summed and passed through an activation.
    With edge_weight, the per-relation mean becomes a weighted mean. Relations in
    sum_relations are a weighted sum instead, for weights normalized outside
    (e.g. D^-1/2 A D^-1/2). msg_dropout drops the aggregated neighbor part
    before the self-loop is added (GLASS drops its messages the same way).
    """

    def __init__(self, in_dim: int, out_dim: int, num_relations: int,
                 self_loop: bool = True, directed: bool = False, sum_relations=(), msg_dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.self_loop = self_loop
        self.directed = directed
        self.sum_relations = set(sum_relations)
        self.msg_dropout = msg_dropout

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
            edge_weight: torch.Tensor = None,  # [E]  optional positive weight per edge
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

            w = torch.ones(src.shape[0], device=x.device) if edge_weight is None else edge_weight[mask]

            if not self.directed:
                src = torch.cat([src, edge_index[1, mask]])
                dst = torch.cat([dst, edge_index[0, mask]])
                w = torch.cat([w, w])

            # Transform each node once, then gather per edge (same result as x[src] @ W, far fewer FLOPs)
            msg = (x @ self.weight[r])[src] * w.unsqueeze(1)

            deg = torch.zeros(N, device=x.device)
            deg.scatter_add_(0, dst, w)
            deg_inv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))

            agg = torch.zeros(N, self.out_dim, device=x.device)
            agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)

            out += agg if r in self.sum_relations else agg * deg_inv.unsqueeze(1)

        if self.msg_dropout > 0:
            out = F.dropout(out, p=self.msg_dropout, training=self.training)

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
            directed: bool = False,
            sum_relations=(),
            norm=None,
            jk=False,
            msg_dropout=0.0
    ):
        """
        norm: None, "batch", "layer" or "graph" - a normalization after every layer (GLASS uses graph).
        jk: return every layer's output concatenated (jumping knowledge, GLASS's default), normalized
            as one; the output size is then out_dim plus hidden_dim per extra layer (self.out_dim).
        msg_dropout: dropout on each layer's aggregated messages (GLASS's dropout inside its conv).
        """
        super().__init__()
        self.dropout = dropout
        self.jk = jk

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.out_dim = sum(dims[1:]) if jk else out_dim
        self.layers = nn.ModuleList([
            RGCNLayer(dims[i], dims[i + 1], num_relations, directed=directed, sum_relations=sum_relations,
                      msg_dropout=msg_dropout)
            for i in range(num_layers)
        ])
        norm_cls = {None: None, "batch": nn.BatchNorm1d, "layer": nn.LayerNorm, "graph": GraphNorm}[norm]
        # With jk the last norm acts on the concatenation instead of the last layer alone
        norm_dims = dims[1:-1] + [self.out_dim]
        self.norms = None if norm_cls is None else nn.ModuleList([norm_cls(d) for d in norm_dims])

    def forward(self, x, edge_index, edge_type, edge_weight=None):
        xs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_type, edge_weight)
            xs.append(x)
            last = i == len(self.layers) - 1
            if self.norms is not None and not (last and self.jk):
                x = self.norms[i](x)
            if not last:  # activation on all but last
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        if self.jk:
            x = torch.cat(xs, dim=1)
            if self.norms is not None:
                x = self.norms[-1](x)
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
    With edge_weight w_ij, log(w_ij) is added to e_{ij}^r, so the attention
    starts from the weights' proportions and learns on top of them.
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
            edge_weight: torch.Tensor = None,  # [E] optional positive weight per edge
    ) -> torch.Tensor:
        N = x.size(0)
        out = torch.zeros(N, self.out_dim, device=x.device)

        for r in range(self.num_relations):
            mask = edge_type == r
            if not mask.any():
                continue

            src = edge_index[0, mask]
            dst = edge_index[1, mask]

            w = None if edge_weight is None else edge_weight[mask]

            if not self.directed:
                src, dst = torch.cat([src, dst]), torch.cat([dst, src])
                w = None if w is None else torch.cat([w, w])

            # Project all nodes once: [N, num_heads, head_dim]. Computing the
            # attention logits from per-node halves (a^T[Wh_i||Wh_j] =
            # a_src^T Wh_i + a_dst^T Wh_j) avoids materializing [E_r, out_dim]
            # tensors for the logits - critical for large edge sets.
            h = (x @ self.weight[r]).view(N, self.num_heads, self.head_dim)
            a_src = self.attn[r][:, :self.head_dim]  # [num_heads, head_dim]
            a_dst = self.attn[r][:, self.head_dim:]
            s_src = (h * a_src).sum(dim=-1)  # [N, num_heads]
            s_dst = (h * a_dst).sum(dim=-1)

            # e: [E_r, num_heads]
            e = self.leaky_relu(s_src[src] + s_dst[dst])
            if w is not None:
                e = e + w.log().unsqueeze(1)

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

            # Weighted aggregation: alpha * h[src], then sum at dst
            # [E_r, num_heads, head_dim]
            weighted = alpha.unsqueeze(-1) * h[src]

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
            norm=None,
            jk=False,
    ):
        """norm, jk: as in RGCN."""
        super().__init__()
        self.dropout = dropout
        self.jk = jk

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.out_dim = sum(dims[1:]) if jk else out_dim
        norm_cls = {None: None, "batch": nn.BatchNorm1d, "layer": nn.LayerNorm, "graph": GraphNorm}[norm]
        norm_dims = dims[1:-1] + [self.out_dim]
        self.norms = None if norm_cls is None else nn.ModuleList([norm_cls(d) for d in norm_dims])
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

    def forward(self, x, edge_index, edge_type, edge_weight=None):
        xs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_type, edge_weight)
            xs.append(x)
            last = i == len(self.layers) - 1
            if self.norms is not None and not (last and self.jk):
                x = self.norms[i](x)
            if not last:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        if self.jk:
            x = torch.cat(xs, dim=1)
            if self.norms is not None:
                x = self.norms[-1](x)
        return x  # logits; apply softmax/sigmoid outside
