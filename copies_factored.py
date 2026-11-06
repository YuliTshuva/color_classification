"""
Yuli Tshuva.
copies_factored.py - the copies graph's outside edges without building them.

In the copies graph (run_gpen_compare.copy_edge_kinds) graph edge u-v joins every
copy of u to every copy of v, except the pairs of one subgraph (the inside edges).
That is up to 1.3B edges on hpo_neuro. But a copy a of node u in subgraph A gets
over its outside edges

    sum over v in N(u) of (sum of all of v's copies)  -  sum over its inside edges

and the first term is the same for every copy of u: one sparse product with the
dataset's own graph. So every row's message is computed exactly, at the cost of
the plain graph plus the inside edges. Mean divides by counts got the same way,
gcn scales by degrees got the same way. Outside edges are split by the receiver
and the sender being in a subgraph ("m") or in none ("p"):
    Osub = mm, Onone = mp + pm + pp, Oin = mm + mp, Oout = pm + pp   (receiver, sender)
Inside (I*), sibling (S) and clique (C) edges are small and stay explicit lists.

Exact for RGCN with mean, sum or gcn aggregation (only the float summation order
differs). Not for attention (RGAT): its weight differs per edge.
"""

# Imports
import torch
import torch.nn.functional as F
from models import RGCNLayer, RGATLayer
from run_gpen_compare import KIND_GROUPS, copy_rows, copy_edge_kinds, build_copy_edges

OUTSIDE_KINDS = ("Osub", "Onone", "Oin", "Oout")
EXPLICIT_KINDS = ("Icopy", "Iplain", "S", "C")
M, P = 0, 1  # row classes: member of a subgraph (a copy or an unshared member), plain (in no subgraph)
CELLS = {"Osub": [(M, M)], "Onone": [(M, P), (P, M), (P, P)], "Oin": [(M, M), (M, P)], "Oout": [(P, M), (P, P)]}


def expand(coloring):
    """The coloring's colors as lists of edge kinds."""
    return [[k for name in color for k in KIND_GROUPS.get(name, (name,))] for color in coloring]


def needs_factoring(hps):
    """Whether hps is a copies coloring with outside edges (the ones too big to build)."""
    return bool(hps.get("coloring")) and bool(hps.get("copies")) \
        and any(k in OUTSIDE_KINDS for color in expand(hps["coloring"]) for k in color)


class FactoredRelations:
    """
    Stands in for rel_edge_index: the relations of a copies coloring, with the
    outside edges kept as the graph plus per-relation cell counts.
    """

    def __init__(self, n_nodes, n_rows, node_of_row, row_class, adj, inside, explicit, cells, agg):
        self.n_nodes, self.n_rows = n_nodes, n_rows
        self.node_of_row, self.row_class = node_of_row, row_class
        self.adj, self.inside, self.explicit, self.cells, self.agg = adj, inside, explicit, cells, agg
        self.rows = [(row_class == c).nonzero(as_tuple=True)[0] for c in (M, P)]
        self.num_relations = len(explicit)
        ones = torch.ones(n_rows, 1, device=adj.device)
        self.in_deg = [self._sum(r, ones).squeeze(1) for r in range(self.num_relations)]
        self.src_scale = self.dst_scale = None
        if agg == "gcn":
            # Every kind is symmetric, so a row's out-degree is its in-degree under the transposed cells
            out_deg = [self._sum(r, ones, transpose=True).squeeze(1) for r in range(self.num_relations)]
            self.src_scale = [torch.where(d > 0, d.rsqrt(), torch.zeros_like(d)) for d in out_deg]
            self.dst_scale = [torch.where(d > 0, d.rsqrt(), torch.zeros_like(d)) for d in self.in_deg]

    def to(self, device):
        moved = object.__new__(FactoredRelations)
        for k, v in vars(self).items():
            if torch.is_tensor(v):
                v = v.to(device)
            elif isinstance(v, list) and v and torch.is_tensor(v[0]):
                v = [t.to(device) for t in v]
            setattr(moved, k, v)
        return moved

    def _sum(self, r, z, transpose=False):
        """Sum of the messages z [n_rows, d] into every row over relation r's edges."""
        out = torch.zeros(self.n_rows, z.size(1), device=z.device, dtype=z.dtype)
        src, dst = self.explicit[r]
        if src.numel():
            out.index_add_(0, dst, z[src])
        cells = self.cells[r].t() if transpose else self.cells[r]
        for recv in (M, P):
            if not cells[recv].any():
                continue
            per_node = torch.zeros(self.n_nodes, z.size(1), device=z.device, dtype=z.dtype)
            for send in (M, P):
                c = int(cells[recv, send])
                if c:
                    rows = self.rows[send]
                    per_node.index_add_(0, self.node_of_row[rows], z[rows] * c if c != 1 else z[rows])
            gathered = torch.sparse.mm(self.adj, per_node)
            rows = self.rows[recv]
            out.index_add_(0, rows, gathered[self.node_of_row[rows]])
        c = int(cells[M, M])
        if c:
            # The pairs of one subgraph were counted above; they are inside edges, not outside ones
            out.index_add_(0, self.inside[1], -c * z[self.inside[0]])
        return out

    def aggregate(self, r, z):
        """Relation r's aggregated messages, as RGCNLayer computes them from the full edge list."""
        if self.agg == "gcn":
            return self._sum(r, z * self.src_scale[r].unsqueeze(1)) * self.dst_scale[r].unsqueeze(1)
        out = self._sum(r, z)
        if self.agg == "sum":
            return out
        deg = self.in_deg[r]
        return out * torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg)).unsqueeze(1)


def build_factored_relations(prep, hps):
    """
    The FactoredRelations of hps' coloring, cached in prep. Same return as
    run_gpen_compare.build_relations, with the FactoredRelations as the edge index.
    """
    agg = hps.get("graph_agg", "mean")
    key = (repr(hps["coloring"]), agg)
    cache = prep.setdefault("factored", {})
    if key not in cache:
        n_nodes, member_node = prep["n_nodes"], prep["member_node"]
        rep, shared = copy_rows(member_node, n_nodes)
        n_rows = n_nodes + int(shared.sum())
        node_of_row = torch.arange(n_rows)
        node_of_row[n_nodes:] = member_node[shared]
        row_class = torch.full((n_rows,), P, dtype=torch.long)
        row_class[member_node[shared].unique()] = -1  # a shared node's own row: replaced by its copies, no edges
        row_class[rep] = M
        if prep["copy_edges"] is None:
            prep["copy_edges"] = build_copy_edges(prep["subgraphs"], prep["graph_edges"], n_nodes)
        inside = rep[prep["copy_edges"][0]]
        inside = torch.cat([inside, inside.flip(0)], dim=1)
        colors = expand(hps["coloring"])
        kinds = {k for color in colors for k in color if k in EXPLICIT_KINDS}
        missing = kinds - set(prep["edge_kinds"])
        if missing:
            prep["edge_kinds"].update(copy_edge_kinds(prep, missing))
        explicit, cells = [], []
        for color in colors:
            lists = [prep["edge_kinds"][k] for k in color if k in EXPLICIT_KINDS]
            explicit.append(torch.cat(lists, dim=1) if lists else torch.empty(2, 0, dtype=torch.long))
            c = torch.zeros(2, 2, dtype=torch.long)
            for k in color:
                for recv, send in CELLS.get(k, []):
                    c[recv, send] += 1
            cells.append(c)
        a, b = prep["graph_edges"]
        adj = torch.sparse_coo_tensor(torch.stack([torch.cat([a, b]), torch.cat([b, a])]),
                                      torch.ones(2 * a.numel()), (n_nodes, n_nodes)).coalesce()
        cache[key] = FactoredRelations(n_nodes, n_rows, node_of_row, row_class, adj, inside, explicit, cells, agg)
    rel = cache[key]
    sum_relations = list(range(rel.num_relations)) if agg in ("sum", "gcn") else []
    return rel, torch.empty(0, dtype=torch.long), rel.num_relations, None, sum_relations


# ---------------------------------------------------------------------------
# Patches: the layers read a FactoredRelations; build_relations makes one
# ---------------------------------------------------------------------------

_rgcn_forward, _rgat_forward = RGCNLayer.forward, RGATLayer.forward


def _factored_rgcn_forward(self, x, edge_index, edge_type, edge_weight=None):
    if not isinstance(edge_index, FactoredRelations):
        return _rgcn_forward(self, x, edge_index, edge_type, edge_weight)
    out = torch.zeros(x.size(0), self.out_dim, device=x.device)
    for r in range(self.num_relations):
        out += edge_index.aggregate(r, x @ self.weight[r])
    if self.msg_dropout > 0:
        out = F.dropout(out, p=self.msg_dropout, training=self.training)
    if self.self_loop:
        out += x @ self.self_weight
    return out


def _factored_rgat_forward(self, x, edge_index, *args, **kwargs):
    if isinstance(edge_index, FactoredRelations):
        raise NotImplementedError("attention is not exact on the factored copies graph")
    return _rgat_forward(self, x, edge_index, *args, **kwargs)


def patch():
    """Route copies colorings with outside edges through the factored relations."""
    import run_gpen_compare as rgc
    original = rgc.build_relations

    def build_relations(prep, hps):
        return build_factored_relations(prep, hps) if needs_factoring(hps) else original(prep, hps)

    rgc.build_relations = build_relations
    RGCNLayer.forward = _factored_rgcn_forward
    RGATLayer.forward = _factored_rgat_forward
