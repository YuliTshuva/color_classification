"""
Yuli Tshuva.
run_gpen_compare.py - our method on the eight subgraph-classification benchmarks
that GPEN (ICML 2025) reports on, scored the way GPEN scores them.

Datasets:
  real      : ppi_bp, hpo_metab, hpo_neuro, em_user  (SubGNN release, data/raw/subgnn/<name>/)
  synthetic : density, cut_ratio, coreness, component (GLASS tmp.npy, data/raw/subgnn_synthetic/<name>/)

What carries over from our method (exp08-exp10):
  - a relational GNN over two relations: the graph, and an edge between every
    pair of nodes that share a subgraph (our same-color relation),
  - a node head trained on the subgraph label, next to the subgraph head,
  - AdamW.
  One study searches every variant at once: the backbone (rgat, rgcn), the
  exp09 readouts (train09.make_readout, so the same modules), and a backbone
  space narrowed from train09/10's by the first POC run (results/gpen_compare):
  multistat and gated attention were the only readouts to win, one or two
  layers, two values each for dropout and weight decay, four attention heads.
  The learning rate is searched instead.

Training (v2, after the first POC run stopped density/cut_ratio at epoch ~31
with training accuracy still at chance):
  - at least MIN_EPOCHS epochs, then stop after PATIENCE epochs without a better
    validation micro-F1 (ties broken by lower validation loss),
  - the checkpoint kept is the one with the best validation micro-F1,
  - the learning rate halves when the validation loss stalls (ReduceLROnPlateau).

What changes to fit the benchmarks:
  - The graph is given, so the first relation is the dataset's edge list instead
    of our text-similarity kNN graph. The exp10 graph axes (hybrid kNN, centered
    or whitened similarity) therefore have nothing to act on and are left out.
  - The splits are the fixed train/val/test splits shipped with the data.
  - n classes: every head outputs one logit per class, softmax + cross-entropy.
    Binary EM-User is the 2-class case, so its micro-F1 is GPEN's accuracy.
  - Multi-label (hpo_neuro): one sigmoid logit per class, i.e. one binary head
    per class run on every subgraph, BCE loss, predicted when the logit > 0.
  - Subgraphs share nodes. Membership is a list of (node, subgraph) pairs and
    every readout pools over those pairs, the same-subgraph relation is the union
    of the per-subgraph cliques, and a supernode links to every member. A node's
    target for the node head is the mean label of the subgraphs of that split
    that contain it, so a node in both a train and a test subgraph only ever sees
    the train label.
  - The node input is a hyperparameter: a learned per-node vector (our
    featureless arm), structural features (degree, neighbors in the node's own
    subgraphs, their fraction, number of subgraphs), or the SubGNN pretrained
    GIN embeddings (real datasets only). The constant input of the first run is
    gone: under mean aggregation it gives every node the same embedding.

Metric: micro-F1 on the subgraph labels, as in GPEN's tables. Selection is on
validation micro-F1 only; test is logged and never optimized on. The best
config is then retrained over 10 seeds and reported as mean +- std (x100).

Usage (server):
  python run_gpen_compare.py                                   # all 7 cases x 8 datasets, one per GPU at a time
  python run_gpen_compare.py --cases transformer extra_node    # only these cases
  python run_gpen_compare.py --cases hub_graph --exclude em_user  # the hub-to-hub relation
  python run_gpen_compare.py --cases clique_sym clique_sym_w --exclude em_user  # clique normalization
  # The run of report 08: the new cases, then pool's seeds again for their test predictions,
  # then the error-against-size plot
  python run_gpen_compare.py --cases hub_graph no_clique clique_sym clique_sym_w --exclude em_user
  python run_gpen_compare.py --cases pool --seeds-only --exclude em_user
  python scripts/plot_error_vs_size.py
  # The run after report 08: fine-tuned GIN input, per-subgraph copies, and split graph edges
  python run_gpen_compare.py --cases gin_ft member_copies edge_label --exclude em_user --max-parallel 3
  python run_gpen_compare.py --jobs-per-gpu 2                 # two jobs share each GPU
  python run_gpen_compare.py --exclude em_user                 # all datasets but em_user
  python run_gpen_compare.py --dataset ppi_bp --smoke          # quick sanity run of every case
  python run_gpen_compare.py --dataset ppi_bp                  # one dataset: study (250 trials) + 10 seeds per case
  python run_gpen_compare.py --dataset em_user --seeds-only    # redo the seeds after a study
  python run_gpen_compare.py --collect                         # rebuild each case's summary.csv only

Results (<dir> is gpen_compare_v2 for pool, gpen_compare_<case> for the others):
  results/<dir>/<dataset>/trial_results.csv
  results/<dir>/<dataset>/hp_results.json
  results/<dir>/<dataset>/seed_results.csv
  results/<dir>/<dataset>/summary.json
  results/<dir>/<dataset>.log   <- console output of each dataset under --dataset all
  results/<dir>/summary.csv     <- one row per dataset; bring these folders back
"""

# Imports
from models import *
from train08 import EPOCHS, LR, SEED, set_seed, normalize_features
from train09 import BACKBONES, make_readout
from sklearn.metrics import f1_score
from os.path import join
import argparse
import csv
import gc
import glob
import hashlib
import itertools
import json
import numpy as np
import optuna
import os
import pandas as pd
import subprocess
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

REAL = ["ppi_bp", "hpo_metab", "hpo_neuro", "em_user"]
SYNTHETIC = ["density", "cut_ratio", "coreness", "component"]
DATASETS = REAL + SYNTHETIC
# By edges + same-subgraph pairs, for scheduling --dataset all
HEAVIEST_FIRST = ["em_user", "hpo_neuro", "hpo_metab", "ppi_bp", "coreness", "component", "cut_ratio", "density"]
PRETRAINED = {"gin": "gin_embeddings.pth"}  # GraphSAINT-GCN trailed GIN and learned vectors in run 1
SPLITS = {"train": 0, "val": 1, "test": 2}
# The three cases - how a subgraph's representation is built:
#   "pool"        : the second relation is the clique of each subgraph; pool the members,
#                   multistat or gated attention (the v2 run)
#   "transformer" : the clique as well; self-attention layers among the members of each
#                   subgraph, then attention pooling with a learned query (a Set Transformer)
#   "extra_node"  : no clique; one extra node per subgraph, linked to its members by the second
#                   relation, and its GNN embedding is the representation (no pooling)
#   "hub_graph"   : extra_node plus a third relation between the extra nodes (hubs), like
#                   SubGNN's graph of subgraphs. Hubs i and j are weighted by the nodes their
#                   subgraphs share plus the graph edges between them; each hub keeps its
#                   HUB_TOP_K strongest links (all links make it near-complete on most datasets)
#   "clique_sym"  : pool with the clique normalized as D^-1/2 A D^-1/2 (D counts clique
#                   neighbors) and summed, instead of each node taking the mean of its clique
#   "clique_sym_w": clique_sym with each clique edge also weighted by 1/N_C (N_C the subgraph
#                   size; summed over the subgraphs a pair shares). Under a mean the 1/N_C
#                   factor would cancel, so it only acts on top of the symmetric sum
#   "no_clique"   : pool without the clique - the graph is the only relation (the baseline of
#                   the error-against-size plot, scripts/plot_error_vs_size.py)
#   "gin_ft"      : pool with SubGNN's pretrained GIN vectors as the starting point of a trainable
#                   input (GLASS fine-tunes its pretrained input), instead of frozen or learned
#                   from scratch. Real datasets only - the synthetic ones ship no vectors
#   "member_copies": the clique replaced by one copy of each node per subgraph it is in. A copy
#                   is linked to its node, and to the copies of the same subgraph that are its
#                   graph neighbors; that inside relation is summed, so a copy can count its
#                   inside edges for its own subgraph. The readout pools the copies, so a node in
#                   two subgraphs no longer blends them (GLASS's in/out marks, exact per subgraph)
#   "edge_label"  : pool with the graph edges split into two relations: inside (the two ends
#                   share a subgraph; summed) and outside (mean). No copies, so a node in two
#                   subgraphs counts both - the control that shows what the copies add
# member_copies and edge_label use RGCN whatever pool's backbone: RGAT's softmax can only average,
# and counting edges is their point.
# Only pool runs the full hyperparameter search. The other ones start from pool's best config
# for the dataset with the readout swapped (the clique cases keep pool's readout), grid-search
# only their small grid (small_grid; one seed per point), then retrain the best point over the seeds.
CASES = ["pool", "transformer", "extra_node", "hub_graph", "clique_sym", "clique_sym_w", "no_clique",
         "gin_ft", "member_copies", "edge_label"]
CASE_READOUTS = {"pool": ["multistat", "gated_attn"],  # the only winners of the first run
                 "transformer": ["transformer"], "extra_node": ["extra_node"], "hub_graph": ["hub_graph"],
                 "clique_sym": None, "clique_sym_w": None, "no_clique": None,  # None: pool's own best readout
                 "gin_ft": None, "member_copies": None, "edge_label": None}
REAL_ONLY_CASES = ("gin_ft",)
COUNTING_CASES = ("member_copies", "edge_label")  # forced to RGCN
# pool is "mean", the default; "none" drops the clique relation
CLIQUE_NORMS = {"clique_sym": "sym", "clique_sym_w": "sym_w", "no_clique": "none"}
SUPERNODE_READOUTS = ("supernode", "extra_node", "hub_graph")  # the color head reads an extra node
TRANSFORMER_HEADS = 4  # must divide gnn_embedding_dim (100, 128, 156)
HUB_TOP_K = 10
NEW_READOUT_HPS = {"transformer": {"pma_heads": TRANSFORMER_HEADS}, "extra_node": {},
                   "hub_graph": {"hub_top_k": HUB_TOP_K}}
SMALL_SEARCH = {"transformer": {"lr": [1e-3, 3e-3], "pma_sabs": [1, 2], "attn_pool_dropout": [0.0, 0.1]},
                "extra_node": {"lr": [1e-3, 3e-3, 1e-2], "supernode_init": ["zeros", "mean"]},
                "hub_graph": {"lr": [1e-3, 3e-3, 1e-2], "supernode_init": ["zeros", "mean"]},
                # A sum changes the scale of the clique messages, so the learning rate is re-searched
                "clique_sym": {"lr": [1e-3, 3e-3, 1e-2]},
                "clique_sym_w": {"lr": [1e-3, 3e-3, 1e-2]},
                "no_clique": {"lr": [1e-3, 3e-3, 1e-2]},
                # GLASS uses dropout 0.5 on the real datasets
                "gin_ft": {"lr": [1e-3, 3e-3, 1e-2], "gnn_dropout_rate": [0.2, 0.5]},
                # copy_clique: also keep our clique, among the copies of each subgraph
                "member_copies": {"lr": [1e-3, 3e-3, 1e-2], "copy_clique": [False, True]},
                "edge_label": {"lr": [1e-3, 3e-3, 1e-2]}}
POOL_ONLY_HPS = ["attn_pool_hidden_dim", "attn_pool_dropout"]  # gated attention's own settings
# Each case gets its own results folder; pool is the v2 run, so it keeps that folder
CASE_DIRS = {"pool": "gpen_compare_v2", "transformer": "gpen_compare_transformer",
             "extra_node": "gpen_compare_extra_node", "hub_graph": "gpen_compare_hub_graph",
             "clique_sym": "gpen_compare_clique_sym", "clique_sym_w": "gpen_compare_clique_sym_w",
             "no_clique": "gpen_compare_no_clique", "gin_ft": "gpen_compare_gin_ft",
             "member_copies": "gpen_compare_member_copies", "edge_label": "gpen_compare_edge_label"}
MIN_EPOCHS, PATIENCE = 200, 150
LR_FACTOR, LR_PATIENCE, MIN_LR = 0.5, 30, 1e-5


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_dataset(name, data_root):
    """
    Read one benchmark into a common form.
    :return: dict with n_nodes, edges [2, E] (each undirected pair once, no
             self-loops), subgraphs (list of node arrays), labels (list of label
             sets), split (np array, 0 train / 1 val / 2 test), features
             ({input name: [n_nodes, d] tensor}).
    """
    features = {}
    if name in SYNTHETIC:
        import networkx  # noqa: F401 - the pickled graph needs it importable
        obj = np.load(join(data_root, "subgnn_synthetic", name, "tmp.npy"), allow_pickle=True).item()
        g = obj["G"]
        nodes = sorted(g.nodes)
        idx = {v: i for i, v in enumerate(nodes)}
        edges = np.array([[idx[a], idx[b]] for a, b in g.edges], dtype=np.int64)
        subgraphs = [np.array([idx[v] for v in s], dtype=np.int64) for s in obj["subG"]]
        labels = [{str(l)} for l in obj["subGLabel"]]
        split = np.asarray(obj["mask"], dtype=np.int64)
        n_nodes = len(nodes)
    else:
        d = join(data_root, "subgnn", name)
        edges = pd.read_csv(join(d, "edge_list.txt"), sep=" ", header=None,
                            usecols=[0, 1]).to_numpy(np.int64)
        sub = pd.read_csv(join(d, "subgraphs.pth"), sep="\t", header=None,
                          names=["nodes", "label", "split"])
        subgraphs = [np.array([int(x) for x in s.split("-")], dtype=np.int64) for s in sub["nodes"]]
        labels = [set(str(l).split("-")) for l in sub["label"]]
        split = sub["split"].map(SPLITS).to_numpy(np.int64)
        n_nodes = int(max(edges.max(), max(s.max() for s in subgraphs))) + 1
        for key, fname in PRETRAINED.items():
            path = join(d, fname)
            if os.path.exists(path):
                # Some of these were saved with requires_grad set
                feat = torch.load(path, map_location="cpu").detach().float()
                n_nodes = max(n_nodes, feat.size(0))
                features[key] = feat
        for key, feat in features.items():
            assert feat.size(0) == n_nodes, f"{name}: {key} has {feat.size(0)} rows, graph has {n_nodes} nodes"

    edges = edges[edges[:, 0] != edges[:, 1]]
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    assert set(np.unique(split)) <= {0, 1, 2}, f"{name}: unexpected split values"
    return {"n_nodes": n_nodes, "edges": torch.from_numpy(edges.T.copy()),
            "subgraphs": [np.unique(s) for s in subgraphs], "labels": labels,
            "split": split, "features": features}


def build_same_subgraph_edges(subgraphs, n_nodes):
    """
    The same-color relation when colors overlap: the union of the cliques of all
    subgraphs, each unordered pair once (RGCN/RGAT symmetrize internally).
    """
    keys = []
    for s in subgraphs:
        if len(s) < 2:
            continue
        s = torch.from_numpy(s)
        i, j = torch.triu_indices(len(s), len(s), offset=1)
        a, b = s[i], s[j]
        keys.append(torch.minimum(a, b) * n_nodes + torch.maximum(a, b))
    if not keys:
        return torch.empty(2, 0, dtype=torch.long)
    keys = torch.unique(torch.cat(keys))
    return torch.stack([keys // n_nodes, keys % n_nodes])


def inside_edge_mask(edges, same_sub_edges, n_nodes):
    """Graph edges whose two ends share at least one subgraph. :return: [E] bool"""
    a, b = edges
    # Both edge lists keep each pair once as (min, max), so their keys match
    same_keys = same_sub_edges[0] * n_nodes + same_sub_edges[1]
    return torch.isin(torch.minimum(a, b) * n_nodes + torch.maximum(a, b), same_keys)


def build_copy_edges(subgraphs, graph_edges, n_nodes):
    """
    member_copies' relations among the copies. Copy r (row r of the membership
    list, i.e. one (node, subgraph) pair) is node n_nodes + r. Rows are grouped
    by subgraph, in the order of `subgraphs`.
    :return: (inside [2, E_in]: copies of one subgraph whose nodes are graph
              neighbors, clique [2, E_cl]: every pair of copies of one subgraph),
              each unordered pair once
    """
    graph_keys = torch.unique(torch.minimum(graph_edges[0], graph_edges[1]) * n_nodes
                              + torch.maximum(graph_edges[0], graph_edges[1]))
    rows_i, rows_j, offset = [], [], 0
    for s in subgraphs:
        k = len(s)
        if k >= 2:
            i, j = torch.triu_indices(k, k, offset=1)
            rows_i.append(i + offset)
            rows_j.append(j + offset)
        offset += k
    if not rows_i:
        empty = torch.empty(2, 0, dtype=torch.long)
        return empty, empty
    ri, rj = torch.cat(rows_i), torch.cat(rows_j)
    member_node = torch.from_numpy(np.concatenate(subgraphs))
    a, b = member_node[ri], member_node[rj]
    is_edge = torch.isin(torch.minimum(a, b) * n_nodes + torch.maximum(a, b), graph_keys)
    clique = torch.stack([ri, rj]) + n_nodes
    return clique[:, is_edge], clique


def structural_features(edges, same_sub_edges, member_node, n_nodes):
    """
    Degree-based node input: log graph degree, log number of graph neighbors that
    share a subgraph with the node, their fraction of the degree, and log number
    of subgraphs the node is in. Membership carries no labels, so every split may
    use it (GLASS marks subgraph nodes the same way). Standardized per column.
    :return: [n_nodes, 4] float tensor
    """
    a, b = edges
    deg = torch.bincount(torch.cat([a, b]), minlength=n_nodes).float()
    inside = inside_edge_mask(edges, same_sub_edges, n_nodes)
    in_deg = torch.bincount(torch.cat([a[inside], b[inside]]), minlength=n_nodes).float()
    n_member = torch.bincount(member_node, minlength=n_nodes).float()
    x = torch.stack([deg.log1p(), in_deg.log1p(), in_deg / deg.clamp(min=1), n_member.log1p()], dim=1)
    return (x - x.mean(0)) / x.std(0).clamp(min=1e-6)


def build_member_pairs(member_sub):
    """
    The transformer's self-attention support, in membership-row space (row r is
    one (node, subgraph) pair): every ordered pair of rows of the same subgraph,
    self-loops included. A node in two subgraphs is two rows, so the attention
    stays block-diagonal over the subgraphs. The rows must be grouped by subgraph.
    """
    pairs, offset = [], 0
    for size in torch.bincount(member_sub).tolist():
        rows = torch.arange(offset, offset + size)
        pairs.append(torch.stack([rows.repeat_interleave(size), rows.repeat(size)]))
        offset += size
    return torch.cat(pairs, dim=1)


def build_clique_weights(subgraphs, same_sub_edges, n_nodes, norm):
    """
    Weights of the same-subgraph edges, in same_sub_edges' order, for a summed
    (not averaged) clique relation. "sym": 1 / sqrt(d_a d_b), d the number of
    clique neighbors (GCN's D^-1/2 A D^-1/2). "sym_w": the same times the sum of
    1/N_C over the subgraphs C that hold both a and b.
    :return: [E] float tensor
    """
    a, b = same_sub_edges
    deg = torch.bincount(torch.cat([a, b]), minlength=n_nodes).float()
    w = (deg[a] * deg[b]).rsqrt()
    if norm == "sym":
        return w
    keys, inv_sizes = [], []
    for s in subgraphs:
        if len(s) < 2:
            continue
        s = torch.from_numpy(s)
        i, j = torch.triu_indices(len(s), len(s), offset=1)
        keys.append(torch.minimum(s[i], s[j]) * n_nodes + torch.maximum(s[i], s[j]))
        inv_sizes.append(torch.full((len(i),), 1.0 / len(s)))
    # Sorted unique keys, so the order matches build_same_subgraph_edges
    uniq, inverse = torch.unique(torch.cat(keys), return_inverse=True)
    assert torch.equal(uniq, a * n_nodes + b)
    return w * torch.zeros(len(uniq)).index_add_(0, inverse, torch.cat(inv_sizes))


def build_hub_edges(graph_edges, member_node, member_sub, n_nodes, n_subs, top_k):
    """
    The hub_graph relation, between the extra nodes. The weight of hubs i and j
    is the number of nodes their subgraphs share plus the number of graph edges
    between them: M (I + A) M^T off the diagonal, M the [subgraph, node]
    membership. Each hub keeps its top_k strongest links; a link kept by either
    end stays, once.
    :return: (edges [2, E] in extra-node ids n_nodes + c, weights [E] float)
    """
    import scipy.sparse as sp
    m = sp.csr_matrix((np.ones(len(member_node)), (member_sub.numpy(), member_node.numpy())),
                      shape=(n_subs, n_nodes))
    a, b = graph_edges.numpy()
    adj = sp.csr_matrix((np.ones(len(a)), (a, b)), shape=(n_nodes, n_nodes))
    w = (m @ (sp.identity(n_nodes, format="csr") + adj + adj.T) @ m.T).toarray()
    np.fill_diagonal(w, 0)
    k = min(top_k, n_subs - 1)
    top = np.argpartition(-w, k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n_subs), k)
    keep = w[rows, top.ravel()] > 0  # no zero-weight links when a hub has fewer than k neighbors
    i, j = rows[keep], top.ravel()[keep]
    pairs = np.unique(np.stack([np.minimum(i, j), np.maximum(i, j)], axis=1), axis=0)
    weights = torch.from_numpy(w[pairs[:, 0], pairs[:, 1]]).float()
    return torch.from_numpy(pairs.T.copy()) + n_nodes, weights


def prepare(ds):
    """Turn a loaded dataset into the tensors training needs. Done once per run."""
    classes = sorted(set().union(*ds["labels"]))
    class_idx = {c: i for i, c in enumerate(classes)}
    n_subs, n_classes, n_nodes = len(ds["subgraphs"]), len(classes), ds["n_nodes"]
    multilabel = any(len(l) > 1 for l in ds["labels"])

    y_hot = torch.zeros(n_subs, n_classes)
    for i, l in enumerate(ds["labels"]):
        y_hot[i, [class_idx[c] for c in l]] = 1.0
    y = y_hot if multilabel else y_hot.argmax(1)

    member_sub = torch.cat([torch.full((len(s),), i, dtype=torch.long)
                            for i, s in enumerate(ds["subgraphs"])])
    member_node = torch.from_numpy(np.concatenate(ds["subgraphs"]))
    split = torch.from_numpy(ds["split"])

    # Node-head targets, per split: the mean label of that split's subgraphs that
    # contain the node. Soft class distribution (multi-class) or per-class
    # frequency (multi-label).
    node_targets, node_masks = {}, {}
    for s in SPLITS.values():
        sel = split[member_sub] == s
        t = torch.zeros(n_nodes, n_classes).index_add_(0, member_node[sel], y_hot[member_sub[sel]])
        cnt = torch.bincount(member_node[sel], minlength=n_nodes)
        node_targets[s] = t / cnt.clamp(min=1).unsqueeze(1).float()
        node_masks[s] = cnt > 0

    same_sub_edges = build_same_subgraph_edges(ds["subgraphs"], n_nodes)
    struct = structural_features(ds["edges"], same_sub_edges, member_node, n_nodes)
    # Supernode of subgraph c is node n_nodes + c, linked to every member
    supernode_edges = torch.stack([member_node, n_nodes + member_sub])

    return {"n_nodes": n_nodes, "n_subs": n_subs, "n_classes": n_classes, "classes": classes,
            "multilabel": multilabel, "y": y, "split": split,
            "member_node": member_node, "member_sub": member_sub,
            "node_targets": node_targets, "node_masks": node_masks,
            "graph_edges": ds["edges"], "same_sub_edges": same_sub_edges,
            "supernode_edges": supernode_edges, "member_pairs": None, "hub_edges": {},
            "copy_edges": None, "graph_inside": None,  # built on first use
            "subgraphs": ds["subgraphs"], "clique_weights": {},
            "features": ds["features"], "struct": struct,
            "stats": {"n_nodes": n_nodes, "n_graph_edges": ds["edges"].size(1),
                      "n_same_subgraph_edges": same_sub_edges.size(1), "n_subgraphs": n_subs,
                      "n_classes": n_classes, "multilabel": multilabel}}


def build_relations(prep, hps):
    """
    Graph first, then the subgraph relation: the cliques, or the edges to the
    extra nodes. train09's supernode readout may keep both, in that order.
    hub_graph adds the weighted hub-to-hub relation last. A clique_norm other
    than "mean" weights the clique edges and makes that relation a sum.
    :return: (rel_edge_index [2, E], rel_edge_type [E], num_relations,
              rel_edge_weight [E] or None when no relation is weighted,
              sum_relations: relation ids RGCN sums instead of averaging)
    """
    readout = hps["readout"]
    if hps.get("copies"):
        return build_copy_relations(prep, hps)
    clique_norm = hps.get("clique_norm", "mean")
    extra_node = readout in SUPERNODE_READOUTS
    rel_edges, rel_weights, sum_relations = [prep["graph_edges"]], [None], []
    if hps.get("graph_split"):
        # edge_label: the graph as two relations, outside (mean) then inside (summed)
        if prep["graph_inside"] is None:
            prep["graph_inside"] = inside_edge_mask(prep["graph_edges"], prep["same_sub_edges"], prep["n_nodes"])
        inside = prep["graph_inside"]
        rel_edges = [prep["graph_edges"][:, ~inside], prep["graph_edges"][:, inside]]
        rel_weights = [None, None]
        sum_relations = [1]
    if clique_norm != "none" and (not extra_node or (readout == "supernode" and hps["keep_same_color"])):
        rel_edges.append(prep["same_sub_edges"])
        if clique_norm == "mean":
            rel_weights.append(None)
        else:
            if clique_norm not in prep["clique_weights"]:
                prep["clique_weights"][clique_norm] = build_clique_weights(
                    prep["subgraphs"], prep["same_sub_edges"], prep["n_nodes"], clique_norm)
            rel_weights.append(prep["clique_weights"][clique_norm])
            sum_relations.append(len(rel_edges) - 1)
    if extra_node:
        rel_edges.append(prep["supernode_edges"])
        rel_weights.append(None)
    if readout == "hub_graph":
        k = hps["hub_top_k"]
        if k not in prep["hub_edges"]:
            prep["hub_edges"][k] = build_hub_edges(prep["graph_edges"], prep["member_node"], prep["member_sub"],
                                                   prep["n_nodes"], prep["n_subs"], k)
        hub_edges, hub_weights = prep["hub_edges"][k]
        rel_edges.append(hub_edges)
        rel_weights.append(hub_weights)
    weights = None
    if any(w is not None for w in rel_weights):
        weights = torch.cat([torch.ones(e.size(1)) if w is None else w for e, w in zip(rel_edges, rel_weights)])
    rel_types = [torch.full((e.size(1),), r, dtype=torch.long) for r, e in enumerate(rel_edges)]
    return torch.cat(rel_edges, dim=1), torch.cat(rel_types), len(rel_edges), weights, sum_relations


def build_copy_relations(prep, hps):
    """
    member_copies' relations, over n_nodes + (number of memberships) nodes: the
    graph (mean), each node to its copies (mean), copies of one subgraph that are
    graph neighbors (summed), and with copy_clique every pair of copies of one
    subgraph (mean). Same return as build_relations.
    """
    n_nodes, n_members = prep["n_nodes"], prep["member_node"].numel()
    if prep["copy_edges"] is None:
        prep["copy_edges"] = build_copy_edges(prep["subgraphs"], prep["graph_edges"], n_nodes)
    inside, clique = prep["copy_edges"]
    copy_links = torch.stack([prep["member_node"], n_nodes + torch.arange(n_members)])
    rel_edges = [prep["graph_edges"], copy_links, inside]
    if hps["copy_clique"]:
        rel_edges.append(clique)
    rel_types = [torch.full((e.size(1),), r, dtype=torch.long) for r, e in enumerate(rel_edges)]
    return torch.cat(rel_edges, dim=1), torch.cat(rel_types), len(rel_edges), None, [2]


def node_input_choices(prep):
    """Node inputs available for a dataset: learned and structural always, pretrained if shipped."""
    return ["learned", "struct"] + sorted(prep["features"])


def build_input(prep, hps):
    """
    :return: (x, module, in_dim). Exactly one of x (a fixed tensor) and module
             (an nn.Embedding whose weight is the input) is not None.
    """
    kind, n = hps["node_input"], prep["n_nodes"]
    if kind == "learned":
        emb = nn.Embedding(n, hps["node_input_dim"])
        nn.init.normal_(emb.weight, std=0.1)
        return None, emb, hps["node_input_dim"]
    if kind == "struct":
        return prep["struct"], None, prep["struct"].size(1)
    # gin_ft: the GIN vectors as the starting point of a trainable input
    pretrained = "gin" if kind == "gin_ft" else kind
    # Pretrained embeddings are unsupervised, so the norm statistics may use every node
    x = normalize_features(prep["features"][pretrained], torch.ones(n, dtype=torch.bool),
                           hps.get("feature_norm", "standardize"))
    if kind == "gin_ft":
        return None, nn.Embedding.from_pretrained(x.clone(), freeze=False), x.size(1)
    return x, None, x.size(1)


def make_gpen_readout(readout, dim, hps):
    """train09's modules: the transformer is its PMA, the extra node its supernode (no module)."""
    base = {"transformer": "pma", "extra_node": "supernode", "hub_graph": "supernode"}.get(readout, readout)
    return make_readout(base, dim, hps)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(prep, hps, epochs=EPOCHS, seed=SEED):
    """
    Train one configuration. The backbone and readout come from hps.
    :return: dict of micro-F1 per split for the subgraph head, plus epochs run.
    """
    set_seed(seed)
    n_classes, n_subs, n_nodes = prep["n_classes"], prep["n_subs"], prep["n_nodes"]
    multilabel, readout, backbone = prep["multilabel"], hps["readout"], hps["backbone"]
    # The color head reads the extra node of each subgraph (train09's supernode)
    supernode = readout in SUPERNODE_READOUTS
    attention_pool = readout in ("pma", "transformer")
    # member_copies: rows n_nodes.. are the copies, one per membership row
    copies = bool(hps.get("copies"))

    feat, feat_module, feat_dim = build_input(prep, hps)
    if copies:
        feat_dim += 1  # the copy flag: 0 on the nodes, 1 on the copies
    rel_edge_index, rel_edge_type, num_relations, rel_edge_weight, sum_relations = build_relations(prep, hps)

    pair_edge_index = None
    if attention_pool and hps["pma_sabs"] > 0:
        if prep["member_pairs"] is None:
            prep["member_pairs"] = build_member_pairs(prep["member_sub"])
        pair_edge_index = prep["member_pairs"].to(DEVICE)

    # --- Modules
    modules = {}
    if feat_module is not None:
        modules["node_input"] = feat_module
    if backbone == "rgcn":
        modules["gnn"] = RGCN(in_dim=feat_dim, hidden_dim=hps["gnn_hidden_dim"],
                              out_dim=hps["gnn_embedding_dim"], num_relations=num_relations,
                              num_layers=hps["k_gnn_layers"], dropout=hps["gnn_dropout_rate"],
                              sum_relations=sum_relations)
    else:
        # RGAT has no sum mode: its softmax renormalizes, so the weights only bias the attention
        modules["gnn"] = RGAT(in_dim=feat_dim, hidden_dim=hps["gnn_hidden_dim"],
                              out_dim=hps["gnn_embedding_dim"], num_relations=num_relations,
                              num_layers=hps["k_gnn_layers"], num_heads=hps["num_heads"],
                              dropout=hps["gnn_dropout_rate"], attn_dropout=hps["gnn_dropout_rate"])
    modules["node_mlp"] = MLP(input_dim=hps["gnn_embedding_dim"], hidden_dims=[hps["gnn_mlp_hidden_dims"]],
                              output_dim=n_classes, dropout_rate=hps["mlp_dropout_rate"])
    pooling = make_gpen_readout(readout, hps["gnn_embedding_dim"], hps)
    if pooling is not None:
        modules["readout"] = pooling
    if readout != "node_score_mean":
        color_in = hps["gnn_embedding_dim"] if supernode else pooling.out_dim
        modules["color_head"] = MLP(input_dim=color_in, hidden_dims=[hps["color_mlp_hidden_dims"]],
                                    output_dim=n_classes, dropout_rate=hps["mlp_dropout_rate"])

    for m in modules.values():
        m.to(DEVICE)
        m.train()
    if feat is not None:
        feat = feat.to(DEVICE)
    rel_edge_index, rel_edge_type = rel_edge_index.to(DEVICE), rel_edge_type.to(DEVICE)
    if rel_edge_weight is not None:
        rel_edge_weight = rel_edge_weight.to(DEVICE)
    member_node, member_sub = prep["member_node"].to(DEVICE), prep["member_sub"].to(DEVICE)
    y, split = prep["y"].to(DEVICE), prep["split"].to(DEVICE)
    node_targets = {s: t.to(DEVICE) for s, t in prep["node_targets"].items()}
    node_masks = {s: m.to(DEVICE) for s, m in prep["node_masks"].items()}
    sub_idx = {s: (split == s).nonzero(as_tuple=True)[0] for s in SPLITS.values()}

    def node_loss(logits, s):
        mask, target = node_masks[s], node_targets[s]
        if multilabel:
            return F.binary_cross_entropy_with_logits(logits[mask], target[mask])
        return -(target[mask] * F.log_softmax(logits[mask], dim=1)).sum(1).mean()

    # node_score_mean produces probabilities rather than logits (as in train09),
    # so its losses take the log / plain BCE of the pooled probabilities.
    def color_loss(out, s):
        idx = sub_idx[s]
        if readout == "node_score_mean":
            p = out[idx].clamp(1e-6, 1 - 1e-6)
            if multilabel:
                return F.binary_cross_entropy(p, y[idx])
            return F.nll_loss(torch.log(p), y[idx])
        if multilabel:
            return F.binary_cross_entropy_with_logits(out[idx], y[idx])
        return F.cross_entropy(out[idx], y[idx])

    params = [p for m in modules.values() for p in m.parameters()]
    optimizer = torch.optim.AdamW(params, lr=hps.get("lr", LR), weight_decay=hps["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=LR_FACTOR,
                                                           patience=LR_PATIENCE, min_lr=MIN_LR)

    def node_x():
        """The GNN input, with the extra node rows appended when the readout needs them."""
        x = feat if feat_module is None else modules["node_input"].weight
        if copies:
            # A copy starts from its node's input; the flag tells the two apart
            x = torch.cat([x, x[member_node]], dim=0)
            flag = torch.zeros(x.size(0), 1, device=x.device)
            flag[n_nodes:] = 1.0
            return torch.cat([x, flag], dim=1)
        if not supernode:
            return x
        if hps["supernode_init"] == "mean":
            sup = scatter_mean(x[member_node], member_sub, n_subs)
        else:
            sup = torch.zeros(n_subs, x.size(1), device=x.device)
        return torch.cat([x, sup], dim=0)

    def forward():
        h = modules["gnn"](node_x(), rel_edge_index, rel_edge_type, rel_edge_weight)
        h_nodes = h[:n_nodes] if supernode or copies else h
        node_logits = modules["node_mlp"](h_nodes)

        if copies:
            # Pool the copies, so each subgraph reads only its own view of a shared node
            pooled = modules["readout"](h[n_nodes:], member_sub, n_subs)
            color_out = modules["color_head"](pooled)
        elif readout == "node_score_mean":
            node_probs = torch.sigmoid(node_logits) if multilabel else torch.softmax(node_logits, dim=1)
            color_out = scatter_mean(node_probs[member_node], member_sub, n_subs)
        elif supernode:
            color_out = modules["color_head"](h[n_nodes:])
        elif attention_pool:
            pooled = modules["readout"](h_nodes[member_node], member_sub, n_subs, pair_edge_index)
            color_out = modules["color_head"](pooled)
        else:
            pooled = modules["readout"](h_nodes[member_node], member_sub, n_subs)
            color_out = modules["color_head"](pooled)
        return node_logits, color_out

    # A probability of 0.5 is a logit of 0
    threshold = 0.5 if readout == "node_score_mean" else 0.0

    def val_f1(color_out):
        """Micro-F1 on the validation subgraphs, on the GPU (accuracy when single-label)."""
        out, target = color_out[sub_idx[1]], y[sub_idx[1]]
        if not multilabel:
            return (out.argmax(1) == target).float().mean().item()
        pred, true = out > threshold, target > 0.5
        tp = (pred & true).sum()
        return (2 * tp / (2 * tp + (pred & ~true).sum() + (~pred & true).sum()).clamp(min=1)).item()

    # --- Training loop. Stop after PATIENCE epochs without a better validation
    # micro-F1 (ties go to the lower validation loss), never before MIN_EPOCHS.
    # Validation is scored in eval mode, so dropout does not blur the selection.
    best_score, unimproved_epochs, best_states = (-1.0, -torch.inf), 0, None
    epoch = 0
    for epoch in range(epochs):
        for m in modules.values():
            m.train()
        node_logits, color_out = forward()
        loss = node_loss(node_logits, 0) + color_loss(color_out, 0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for m in modules.values():
            m.eval()
        with torch.no_grad():
            node_logits, color_out = forward()
            val_loss = (node_loss(node_logits, 1) + color_loss(color_out, 1)).item()
            score = (val_f1(color_out), -val_loss)
        scheduler.step(val_loss)

        if score > best_score:
            best_score = score
            unimproved_epochs = 0
            best_states = {name_: {k: v.detach().clone() for k, v in m.state_dict().items()}
                           for name_, m in modules.items()}
        else:
            unimproved_epochs += 1
        if epoch + 1 >= MIN_EPOCHS and unimproved_epochs >= PATIENCE:
            break

    # --- Evaluate with the best checkpoint
    if best_states is not None:
        for name_, m in modules.items():
            m.load_state_dict(best_states[name_])
    for m in modules.values():
        m.eval()
    with torch.no_grad():
        _, color_out = forward()

    if multilabel:
        pred = (color_out > threshold).long().cpu().numpy()
        true = y.long().cpu().numpy()
    else:
        pred = color_out.argmax(1).cpu().numpy()
        true = y.cpu().numpy()

    out = {"epochs": epoch + 1}
    for split_name, s in SPLITS.items():
        idx = sub_idx[s].cpu().numpy()
        out[f"{split_name}_f1"] = float(f1_score(true[idx], pred[idx], average="micro", zero_division=0))

    # Each test subgraph's labels, as class names ("-"-joined when multi-label),
    # for the error-against-size plot
    def names(labels):
        if multilabel:
            return ["-".join(prep["classes"][c] for c in np.flatnonzero(row)) for row in labels]
        return [prep["classes"][c] for c in labels]
    test = sub_idx[SPLITS["test"]].cpu().numpy()
    out["test_predictions"] = {"subgraph": test, "true": names(true[test]), "pred": names(pred[test])}
    return out


# ---------------------------------------------------------------------------
# Study and seeds
# ---------------------------------------------------------------------------

def sample_hps(trial, input_choices, backbones, readouts):
    """
    train09/train10's backbone space narrowed by the first POC run, with the
    backbone, readout, node input and learning rate as axes, and each readout's
    own hyperparameters as train09 samples them. Fixed values are kept in hps
    (not sampled) so every trial row still records them.
    """
    hps = {
        "backbone": trial.suggest_categorical("backbone", backbones),
        "readout": trial.suggest_categorical("readout", readouts),
        "lr": trial.suggest_categorical("lr", [1e-3, 3e-3, 1e-2]),
        "gnn_embedding_dim": trial.suggest_categorical("gnn_embedding_dim", [100, 128, 156]),
        "gnn_hidden_dim": trial.suggest_categorical("gnn_hidden_dim", [100, 128, 156]),
        "k_gnn_layers": trial.suggest_categorical("k_gnn_layers", [1, 2]),
        "gnn_dropout_rate": trial.suggest_categorical("gnn_dropout_rate", [0.0, 0.2]),
        "gnn_mlp_hidden_dims": trial.suggest_categorical("gnn_mlp_hidden_dims", [50, 64, 80]),
        "mlp_dropout_rate": trial.suggest_categorical("mlp_dropout_rate", [0.0, 0.2]),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-4]),
        "node_input": trial.suggest_categorical("node_input", input_choices),
    }
    if hps["node_input"] == "learned":
        hps["node_input_dim"] = trial.suggest_categorical("node_input_dim", [64, 128])
    elif hps["node_input"] in PRETRAINED:
        hps["feature_norm"] = "standardize"  # the only norm that worked for GIN in run 1
    if hps["backbone"] == "rgat":
        hps["num_heads"] = 4  # won 5 of 7 datasets in run 1

    readout = hps["readout"]
    if readout != "node_score_mean":
        hps["color_mlp_hidden_dims"] = trial.suggest_categorical("color_mlp_hidden_dims", [32, 50, 64])
    if readout == "gated_attn":
        hps["attn_pool_hidden_dim"] = trial.suggest_categorical("attn_pool_hidden_dim", [32, 64, 128])
        hps["attn_pool_dropout"] = trial.suggest_categorical("attn_pool_dropout", [0.0, 0.1, 0.2])
    if readout == "pma":
        hps["pma_heads"] = trial.suggest_categorical("pma_heads", [2, 4])
        hps["pma_sabs"] = trial.suggest_categorical("pma_sabs", [0, 1])
        hps["attn_pool_dropout"] = trial.suggest_categorical("attn_pool_dropout", [0.0, 0.1, 0.2])
    if readout == "supernode":
        hps["keep_same_color"] = trial.suggest_categorical("keep_same_color", [True, False])
    if readout == "supernode":
        hps["supernode_init"] = trial.suggest_categorical("supernode_init", ["zeros", "mean"])
    return hps


TRIAL_FIELDS = ["dataset", "backbone", "readout", "trial", "seed", "train_f1", "val_f1", "test_f1",
                "epochs", "seconds", "hps"]


def append_row(csv_path, row):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRIAL_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def make_row(name, hps, trial, seed, res, seconds):
    return {"dataset": name, "backbone": hps["backbone"], "readout": hps["readout"],
            "trial": trial, "seed": seed,
            "train_f1": round(res["train_f1"], 4), "val_f1": round(res["val_f1"], 4),
            "test_f1": round(res["test_f1"], 4), "epochs": res["epochs"],
            "seconds": round(seconds, 1), "hps": json.dumps(hps)}


def is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def free_memory():
    """Drop what the last trial left behind, so the next one starts from an empty GPU."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_study(name, prep, trials, out_dir, epochs, backbones, readouts, study_name=None):
    """
    Resumable: every finished trial is stored in study.db (an Optuna SQLite
    study), so a rerun after a timeout loads it, keeps the sampler's history and
    only runs the trials still missing. A trial cut off mid-training is redone.
    """
    csv_path, hp_path = join(out_dir, "trial_results.csv"), join(out_dir, "hp_results.json")
    hp_tried = {}
    if os.path.exists(hp_path):
        with open(hp_path) as f:
            hp_tried = json.load(f)
    input_choices = node_input_choices(prep)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                storage=f"sqlite:///{os.path.abspath(join(out_dir, 'study.db'))}",
                                study_name=study_name or name, load_if_exists=True)
    finished = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    done = len(study.get_trials(deepcopy=False, states=finished))
    if done >= trials:
        print(f"[{name}] study already has {done}/{trials} trials - skipping to the seeds")
        return
    if done:
        print(f"[{name}] resuming the study at {done}/{trials} trials")

    def objective(trial):
        hps = sample_hps(trial, input_choices, backbones, readouts)
        if str(hps) in hp_tried:
            return hp_tried[str(hps)]
        print(f"[{name}] trial {trial.number}: {hps}")
        start = time.time()
        try:
            res = train(prep, hps, epochs=epochs)
        except RuntimeError as e:
            if not is_oom(e):
                raise
            # Record the config so it is never retried, prune the trial
            free_memory()
            print(f"[{name}] OOM - pruning trial {trial.number}")
            hp_tried[str(hps)] = 0.0
            with open(hp_path, "w") as f:
                json.dump(hp_tried, f, indent=4)
            raise optuna.TrialPruned()

        append_row(csv_path, make_row(name, hps, trial.number, SEED, res, time.time() - start))
        # Selection on validation micro-F1 only; test is logged, never optimized on
        hp_tried[str(hps)] = res["val_f1"]
        with open(hp_path, "w") as f:
            json.dump(hp_tried, f, indent=4)
        print(f"[{name}] val micro-F1 {res['val_f1']:.4f} ({res['epochs']} epochs)\n" + "*" * 80)
        return res["val_f1"]

    study.optimize(objective, n_trials=trials - done, callbacks=[lambda *_: free_memory()])


def best_hps_of(out_dir):
    """The study's best config by validation micro-F1, from its trial CSV."""
    df = pd.read_csv(join(out_dir, "trial_results.csv"))
    return json.loads(df.groupby("hps")["val_f1"].mean().idxmax())


def pool_best_hps(name, case):
    """
    Pool's best config for the dataset adapted to the case: the readout swapped,
    or for the clique cases pool's readout kept and the clique normalization
    set. None if pool has no result.
    """
    path = join(case_dir("pool"), name, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        hps = json.loads(json.load(f)["hps"])
    if case in CLIQUE_NORMS:
        return {**hps, "clique_norm": CLIQUE_NORMS[case]}
    if case == "gin_ft":
        hps = {k: v for k, v in hps.items() if k != "node_input_dim"}
        return {**hps, "node_input": "gin_ft", "feature_norm": "standardize"}
    if case == "member_copies":
        return {**hps, "backbone": "rgcn", "copies": True}
    if case == "edge_label":
        return {**hps, "backbone": "rgcn", "graph_split": True}
    readout = CASE_READOUTS[case][0]
    hps = {k: v for k, v in hps.items() if k not in POOL_ONLY_HPS}
    return {**hps, "readout": readout, **NEW_READOUT_HPS[readout]}


def small_grid(case, name, base_hps=None):
    """
    The case's grid for this dataset. On the real datasets member_copies also
    tries the fine-tuned GIN input next to pool's own input (base_hps; None when
    only the grid size is needed).
    """
    grid = dict(SMALL_SEARCH[case])
    if case == "member_copies" and name in REAL:
        own = base_hps["node_input"] if base_hps else "pool"
        grid["node_input"] = [own] if own == "gin_ft" else [own, "gin_ft"]
    return grid


def run_small_search(name, prep, out_dir, epochs, base_hps, grid):
    """
    Every point of `grid` on top of base_hps, one seed each, into trial_results.csv.
    Resumable: points already in the CSV are not rerun. An OOM point is recorded
    with zero scores so it is never retried.
    :return: the grid point with the best validation micro-F1
    """
    csv_path = join(out_dir, "trial_results.csv")
    done = {}
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        done = dict(zip(df["hps"], df["val_f1"]))
    points = [{**base_hps, **dict(zip(grid, values))} for values in itertools.product(*grid.values())]
    for i, hps in enumerate(points):
        if json.dumps(hps) in done:
            continue
        print(f"[{name}] grid point {i + 1}/{len(points)}: {hps}")
        start = time.time()
        try:
            res = train(prep, hps, epochs=epochs)
        except RuntimeError as e:
            if not is_oom(e):
                raise
            print(f"[{name}] OOM - recording grid point {i + 1} as 0")
            res = {"train_f1": 0.0, "val_f1": 0.0, "test_f1": 0.0, "epochs": 0}
        free_memory()
        append_row(csv_path, make_row(name, hps, i, SEED, res, time.time() - start))
        done[json.dumps(hps)] = res["val_f1"]
        print(f"[{name}] val micro-F1 {res['val_f1']:.4f} ({res['epochs']} epochs)")
    # Only this grid's points compete, in case the CSV holds points of an older grid
    return max(points, key=lambda hps: done[json.dumps(hps)])


def run_seeds(name, prep, seeds, out_dir, epochs, hps=None):
    """
    Retrain a config over `seeds` seeds and write summary.json: the given hps, or
    the study's best. Each seed's test predictions go to test_predictions.csv
    (one row per test subgraph: its size, true and predicted labels, and whether
    they match exactly). Resumable: seeds already in seed_results.csv and in
    test_predictions.csv for this same config are kept, not rerun; a seed from
    before predictions were saved is rerun.
    """
    hps = hps or best_hps_of(out_dir)
    print(f"[{name}] hps: {hps}")
    csv_path = join(out_dir, "seed_results.csv")
    preds_path = join(out_dir, "test_predictions.csv")
    hps_id = hashlib.md5(json.dumps(hps).encode()).hexdigest()[:10]  # ties predictions to the config

    pred_seeds = set()
    if os.path.exists(preds_path):
        preds = pd.read_csv(preds_path, keep_default_na=False)
        kept = preds[(preds["hps_id"] == hps_id) & (preds["seed"] < seeds)]
        pred_seeds = set(kept["seed"])
        if len(kept) < len(preds):
            kept.to_csv(preds_path, index=False)

    rows = []
    if os.path.exists(csv_path):
        old = pd.read_csv(csv_path)
        # Only rows of the current best config count; a different best means a fresh start
        old = old[old["hps"] == json.dumps(hps)].drop_duplicates("seed")
        rows = [r for r in old.to_dict("records") if r["seed"] < seeds and r["seed"] in pred_seeds]
        if len(rows) < len(pd.read_csv(csv_path)):
            os.remove(csv_path)
            for r in rows:
                append_row(csv_path, r)
        if rows:
            print(f"[{name}] resuming the seeds at {len(rows)}/{seeds}")
    done_seeds = {r["seed"] for r in rows}

    for seed in range(seeds):
        if seed in done_seeds:
            continue
        start = time.time()
        res = train(prep, hps, epochs=epochs, seed=seed)
        row = make_row(name, hps, "best", seed, res, time.time() - start)
        p = res["test_predictions"]
        pd.DataFrame({"hps_id": hps_id, "seed": seed, "subgraph": p["subgraph"],
                      "size": [len(prep["subgraphs"][i]) for i in p["subgraph"]],
                      "true": p["true"], "pred": p["pred"],
                      "correct": [int(t == q) for t, q in zip(p["true"], p["pred"])]}) \
            .to_csv(preds_path, mode="a", header=not os.path.exists(preds_path), index=False)
        append_row(csv_path, row)
        rows.append(row)
        free_memory()
        print(f"[{name}] seed {seed}: val {res['val_f1']:.4f}, test {res['test_f1']:.4f}")

    test = np.array([r["test_f1"] for r in rows]) * 100
    val = np.array([r["val_f1"] for r in rows]) * 100
    trials_path = join(out_dir, "trial_results.csv")
    trials_df = pd.read_csv(trials_path) if os.path.exists(trials_path) else pd.DataFrame()
    # Best validation score reached by each variant, to see which ones the study favoured
    by_variant = [] if trials_df.empty else (
        trials_df.groupby(["backbone", "readout"])["val_f1"].agg(["max", "count"])
        .round(4).reset_index().to_dict("records"))
    summary = {"dataset": name, "backbone": hps["backbone"], "readout": hps["readout"],
               "node_input": hps["node_input"], "metric": "micro-F1 (x100)",
               "test_mean": round(float(test.mean()), 2), "test_std": round(float(test.std()), 2),
               "val_mean": round(float(val.mean()), 2), "val_std": round(float(val.std()), 2),
               "n_seeds": seeds, "n_trials": len(trials_df),
               "mean_seconds_per_run": round(float(np.mean([r["seconds"] for r in rows])), 1),
               **prep["stats"], "hps": json.dumps(hps), "by_variant": json.dumps(by_variant)}
    with open(join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    print(f"[{name}] {hps['backbone']}/{hps['readout']}: "
          f"test micro-F1 {summary['test_mean']:.1f} +- {summary['test_std']:.1f}")


def collect(results_dir):
    """Rebuild <results_dir>/summary.csv from every summary.json under it."""
    rows = []
    for path in sorted(glob.glob(join(results_dir, "*", "summary.json"))):
        with open(path) as f:
            rows.append(json.load(f))
    if not rows:
        print(f"No summary.json under {results_dir}")
        return
    df = pd.DataFrame(rows)
    df["dataset"] = pd.Categorical(df["dataset"], DATASETS, ordered=True)
    df = df.sort_values("dataset")
    tmp = join(results_dir, "summary.csv.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, join(results_dir, "summary.csv"))  # parallel jobs never see a half-written file
    print(df[["dataset", "backbone", "readout", "test_mean", "test_std", "n_seeds", "n_trials"]]
          .to_string(index=False))


def case_dir(case, smoke=False):
    return join("results", CASE_DIRS[case]) + ("_smoke" if smoke else "")


def run_all(args):
    """
    Every (case, dataset) job, spread over all visible GPUs. Each job is a child
    process pinned to one GPU through CUDA_VISIBLE_DEVICES (DEVICE is fixed at
    import, so a process cannot switch GPUs itself). Heaviest datasets go first
    so the long ones start early and the small ones fill the gaps. Without a GPU
    the jobs run one after another on the CPU.
    """
    passthrough = ["--data-root", args.data_root, "--trials", str(args.trials), "--seeds", str(args.seeds),
                   "--epochs", str(args.epochs), "--backbones", *args.backbones]
    if args.seeds_only:
        passthrough.append("--seeds-only")
    if args.smoke:
        passthrough.append("--smoke")

    n_gpus = torch.cuda.device_count()
    # Round-robin over the GPUs, so a cap below the GPU count still uses one GPU per job
    slots = [str(g) for _ in range(args.jobs_per_gpu) for g in range(n_gpus)][:args.max_parallel] or [None]

    def finished(case, name):
        path = join(case_dir(case, args.smoke), name, "summary.json")
        # Runs from before predictions were saved are not finished
        if not os.path.exists(path) or not os.path.exists(join(os.path.dirname(path), "test_predictions.csv")):
            return False
        with open(path) as f:
            s = json.load(f)
        if args.seeds_only:
            return s["n_seeds"] >= args.seeds
        trials = args.trials if case == "pool" else \
            int(np.prod([len(v) for v in small_grid(case, name).values()]))
        return s["n_seeds"] >= args.seeds and s["n_trials"] >= trials

    # Each job is its own process, so all of its memory (RAM and GPU) is
    # returned to the system the moment it ends, before the next one starts.
    jobs = [(case, name) for name in HEAVIEST_FIRST if name not in args.exclude for case in args.cases
            if not (case in REAL_ONLY_CASES and name not in REAL)]
    queue = [job for job in jobs if not finished(*job)]
    skipped = [f"{case}/{name}" for case, name in jobs if (case, name) not in queue]
    if skipped:
        print(f"Already finished, skipping: {skipped}")
    running, failed = {}, []
    for case in args.cases:
        os.makedirs(case_dir(case, args.smoke), exist_ok=True)
    print(f"Running {len(queue)} jobs on {n_gpus} GPU(s), {len(slots)} at a time")

    while queue or running:
        # Fill every free slot
        # Slots are told apart by position: with --jobs-per-gpu > 1 a GPU id repeats
        busy = {i for i, _, _, _, _ in running.values()}
        for i in [i for i in range(len(slots)) if i not in busy]:
            if not queue:
                break
            slot = slots[i]
            case, name = queue.pop(0)
            env = dict(os.environ)
            if slot is not None:
                env["CUDA_VISIBLE_DEVICES"] = slot
            log = open(join(case_dir(case, args.smoke), f"{name}.log"), "w")
            proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--dataset", name,
                                     "--cases", case, *passthrough],
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
            running[proc.pid] = (i, slot, proc, log, f"{case}/{name}")
            print(f"  started {case}/{name} on GPU {slot} (log: {log.name})")
            busy.add(i)

        time.sleep(5)
        for pid, (i, slot, proc, log, job) in list(running.items()):
            if proc.poll() is None:
                continue
            log.close()
            print(f"  finished {job} on GPU {slot} (exit code {proc.returncode})")
            if proc.returncode != 0:
                failed.append(job)
            del running[pid]

    for case in args.cases:
        print(f"\n{case}:")
        collect(case_dir(case, args.smoke))
    if failed:
        print(f"FAILED: {failed} - see their .log files under results/")


def main():
    parser = argparse.ArgumentParser(description="Our method on the GPEN benchmarks.")
    parser.add_argument("--dataset", default="all", choices=DATASETS + ["all"],
                        help="One dataset, or all eight spread over the GPUs (default)")
    parser.add_argument("--jobs-per-gpu", type=int, default=1,
                        help="With --dataset all: datasets sharing one GPU at a time")
    parser.add_argument("--exclude", nargs="+", default=[], choices=DATASETS,
                        help="Datasets to leave out of --dataset all")
    parser.add_argument("--max-parallel", type=int, default=2,
                        help="With --dataset all: datasets running at once, whatever the GPU count. "
                             "Each one holds ~3 GB of RAM, so 2 fits a 10G job")
    parser.add_argument("--backbones", nargs="+", default=BACKBONES, choices=BACKBONES)
    parser.add_argument("--cases", nargs="+", default=CASES, choices=CASES,
                        help="Cases to run (default: all seven)")
    parser.add_argument("--data-root", default=join("data", "raw"),
                        help="Folder holding subgnn/<name>/ and subgnn_synthetic/<name>/")
    parser.add_argument("--trials", type=int, default=250)
    parser.add_argument("--seeds", type=int, default=10, help="GPEN reports 10 runs")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Epoch cap. The default relies on early stopping.")
    parser.add_argument("--seeds-only", action="store_true", help="Skip the study; rerun the seeds")
    parser.add_argument("--collect", action="store_true", help="Only rebuild summary.csv")
    parser.add_argument("--smoke", action="store_true",
                        help="1 trial per readout, 2 seeds, 15 epochs, into results/<case dir>_smoke/")
    args = parser.parse_args()

    if args.collect:
        for case in args.cases:
            print(f"\n{case}:")
            collect(case_dir(case, args.smoke))
        return
    if args.dataset == "all":
        run_all(args)
        return

    start = time.time()
    prep = prepare(load_dataset(args.dataset, args.data_root))
    print(f"[{args.dataset}] loaded in {time.time() - start:.1f}s: {prep['stats']}, "
          f"node inputs {node_input_choices(prep)}, device {DEVICE}")

    for case in args.cases:
        readouts = CASE_READOUTS[case]
        results_dir = case_dir(case, args.smoke)
        out_dir = join(results_dir, args.dataset)
        print(f"[{args.dataset}] case {case}")
        if case in REAL_ONLY_CASES and args.dataset not in REAL:
            print(f"[{args.dataset}] {case} needs pretrained vectors - skipping")
            continue
        if case != "pool":
            # Pool's best config with this case's readout, then the small grid on top of it
            hps = pool_best_hps(args.dataset, case)
            if hps is None:
                print(f"[{args.dataset}] no pool result to take the config from - skipping {case}")
                continue
            os.makedirs(out_dir, exist_ok=True)
            seeds, epochs = (2, 15) if args.smoke else (args.seeds, args.epochs)
            if not args.seeds_only:
                hps = run_small_search(args.dataset, prep, out_dir, epochs, hps,
                                       small_grid(case, args.dataset, hps))
            else:
                hps = best_hps_of(out_dir)
            run_seeds(args.dataset, prep, seeds, out_dir, epochs, hps)
            collect(results_dir)
            continue
        os.makedirs(out_dir, exist_ok=True)
        if args.smoke:
            # Every readout once, on both backbones in turn, so each code path runs.
            # Each gets its own study: Optuna refuses a choice list that changes within one.
            for i, readout in enumerate(readouts):
                run_study(args.dataset, prep, 1, out_dir, 15, [args.backbones[i % len(args.backbones)]],
                          [readout], study_name=f"{args.dataset}_smoke_{readout}")
            run_seeds(args.dataset, prep, 2, out_dir, 15)
        else:
            if not args.seeds_only:
                run_study(args.dataset, prep, args.trials, out_dir, args.epochs, args.backbones, readouts)
            run_seeds(args.dataset, prep, args.seeds, out_dir, args.epochs)
        collect(results_dir)


if __name__ == "__main__":
    main()
