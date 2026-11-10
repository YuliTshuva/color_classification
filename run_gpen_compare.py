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
  # Run 6: copies with inside/outside counts, the transformer over the copies, frozen GIN + learned input
  python run_gpen_compare.py --cases copies_plus --exclude em_user --max-parallel 3
  # Run 7: counts past the GNN (needs copies_plus' results), and GLASS's GNN-plain rebuilt
  python run_gpen_compare.py --cases copies_skip plain_gnn --exclude em_user --max-parallel 3
  # Run 8: the two run-7 cases trained the GLASS way (needs their results)
  python run_gpen_compare.py --cases plain_gnn_fix copies_skip_fix --exclude em_user --max-parallel 3
  # Run 9: GLASS's own pretrained vectors as input (data/raw/subgnn/<name>/glass_embeddings.pth)
  python run_gpen_compare.py --cases plain_gnn_glass copies_glass edge_label_glass copies_skip_tune \
      --exclude em_user --max-parallel 3
  # Run 10: GLASS's training recipe and GLASS's splits (GLASS_EVAL_CASES)
  python run_gpen_compare.py --cases glass_plain edge_label_recipe copies_recipe --max-parallel 3
  python run_gpen_compare.py --jobs-per-gpu 2                # two jobs share each GPU
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
# GraphSAINT-GCN trailed GIN and learned vectors in run 1. glass: GLASS's own pretrained vectors
# (github.com/Xi-yuanWang/GLASS, Emb/<name>_64.pt, same node ids), real datasets
PRETRAINED = {"gin": "gin_embeddings.pth", "glass": "glass_embeddings.pth"}
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
#   "member_copies": the clique replaced by one copy of each shared node (a node in two or more
#                   subgraphs) per subgraph it is in; a node in one subgraph is its own copy. A copy
#                   is linked to its node, and to the copies of the same subgraph that are its
#                   graph neighbors; that inside relation is summed, so a copy can count its
#                   inside edges for its own subgraph. The readout pools the copies, so a node in
#                   two subgraphs no longer blends them (GLASS's in/out marks, exact per subgraph).
#                   copy_all: a copy of every member, as in runs 5-10
#   "edge_label"  : pool with the graph edges split into two relations: inside (the two ends
#                   share a subgraph; summed) and outside (mean). No copies, so a node in two
#                   subgraphs counts both - the control that shows what the copies add
#   "copies_plus" : member_copies without the copy clique, plus each copy's exact inside and
#                   outside edge counts as input (links to the outside neighbors would be ~100M
#                   edges on hpo). The grid tries pool's readout and the transformer over the
#                   copies, and on the real datasets the frozen GIN vectors concatenated with a
#                   learned vector (gin_cat) next to pool's own input
#   "copies_skip" : copies_plus' best config, plus each subgraph's own counts (size, inside
#                   edges, outside edges) fed straight to the color head, past the GNN
#   "plain_gnn"   : GLASS's GNN-plain rebuilt: the graph only, fine-tuned pretrained input,
#                   hidden 64, dropout 0.5, no node head; aggregation mean or GCN-normalized sum.
#                   Real datasets only. Unlike GLASS it trains full-batch, with no norm layers
#   "plain_gnn_fix", "copies_skip_fix": the best config of plain_gnn / copies_skip, trained the
#                   GLASS way: a norm layer after every GNN layer, and the train subgraphs split
#                   into mini-batches, one optimizer step each (the GNN still sees the whole graph)
#   "plain_gnn_glass": plain_gnn_fix's best config with GLASS's own pretrained vectors as the
#                   fine-tuned input (glass_ft) instead of SubGNN's GIN vectors
#   "copies_glass", "edge_label_glass": our two best models on the real datasets (copies_skip_fix,
#                   edge_label) with the glass_ft input; edge_label also tries the two norms
#   "copies_skip_tune": copies_skip_fix's best config with a wider grid (lr, layers, dropout,
#                   norm). Synthetic datasets only - their runs take seconds
# Run 10 - GLASS's training recipe (glass_recipe): mini-batches of GLASS's batch size, its
# learning rate with the decay driven by the training loss, GraphNorm on the input and after
# every layer, dropout on the input and on the messages, its pooling (sum, or size-normalized
# sum) into a single linear layer, and its early stopping. These cases are also scored on
# GLASS's splits (GLASS_EVAL_CASES, glass_split): val and test swapped when val is the smaller
# (ppi_bp), and on the synthetic datasets a fresh random 50/25/25 split per seed, the same
# splits GLASS's code draws.
#   "glass_plain" : GLASS's GNN-plain rebuilt faithfully: the graph only, GLASS's own vectors
#                   as fine-tuned input, GLASS's config file for the dataset (layers, aggregation,
#                   dropout, pooling, jumping knowledge), and the recipe. Real datasets only
#   "edge_label_recipe": edge_label's best config trained with the recipe; the grid tries
#                   its own readout or GLASS's pooling, and its own input or GLASS's vectors
#   "copies_recipe": copies_skip_tune's best config with the rest of the recipe fixed and two
#                   grid axes: own or GLASS's batch size, own readout or GLASS's pooling. Synthetic only
# member_copies, edge_label, copies_plus, copies_skip and copies_skip_fix use RGCN whatever pool's backbone: RGAT's softmax can only average,
# and counting edges is their point.
# Only pool runs the full hyperparameter search. The other ones start from pool's best config
# for the dataset with the readout swapped (the clique cases keep pool's readout), grid-search
# only their small grid (small_grid; one seed per point), then retrain the best point over the seeds.
CASES = ["pool", "transformer", "extra_node", "hub_graph", "clique_sym", "clique_sym_w", "no_clique",
         "gin_ft", "member_copies", "edge_label", "copies_plus", "copies_skip", "plain_gnn",
         "plain_gnn_fix", "copies_skip_fix", "plain_gnn_glass", "copies_glass", "edge_label_glass",
         "copies_skip_tune", "glass_plain", "edge_label_recipe", "copies_recipe"]
GLASS_INPUT_CASES = ("plain_gnn_glass", "copies_glass", "edge_label_glass")
RECIPE_CASES = ("glass_plain", "edge_label_recipe", "copies_recipe")
GLASS_EVAL_CASES = RECIPE_CASES  # scored on GLASS's splits (glass_split)
# GLASS's config/<name>.yml (github.com/Xi-yuanWang/GLASS). resi is the lr decay factor;
# component has none, so it takes test()'s default 0.7. Its code also defaults jk to on.
GLASS_CONFIGS = {
    "ppi_bp": {"hidden_dim": 64, "aggr": "mean", "batch_size": 80, "conv_layer": 2, "dropout": 0.5, "lr": 5e-4, "pool": "sum",
               "resi": 0.2},
    "hpo_metab": {"hidden_dim": 64, "aggr": "gcn", "batch_size": 59, "conv_layer": 1, "dropout": 0.5, "lr": 1e-3, "pool": "sum",
                  "resi": 0.2},
    "hpo_neuro": {"hidden_dim": 64, "aggr": "gcn", "batch_size": 99, "conv_layer": 2, "dropout": 0.5, "lr": 2e-3, "pool": "sum",
                  "resi": 0.2},
    "em_user": {"hidden_dim": 64, "aggr": "gcn", "batch_size": 6, "conv_layer": 1, "dropout": 0.5, "lr": 1e-3, "pool": "size",
                "resi": 0.7},
    "density": {"hidden_dim": 8, "aggr": "sum", "batch_size": 2, "conv_layer": 1, "dropout": 0.0, "lr": 1e-3, "pool": "size",
                "resi": 0.9},
    "cut_ratio": {"hidden_dim": 8, "aggr": "sum", "batch_size": 3, "conv_layer": 1, "dropout": 0.3, "lr": 1e-2, "pool": "mean",
                  "resi": 0.9},
    "coreness": {"hidden_dim": 20, "aggr": "sum", "batch_size": 2, "conv_layer": 2, "dropout": 0.05, "lr": 6.7e-3, "pool": "mean",
                 "resi": 0.9},
    "component": {"hidden_dim": 17, "aggr": "sum", "batch_size": 8, "conv_layer": 1, "dropout": 0.0, "lr": 1e-3, "pool": "sum",
                  "resi": 0.7}}
GLASS_HIDDEN = 64  # GLASS's hidden size on the real datasets, the size of its pretrained vectors
# GLASS's ReduceLROnPlateau (default patience 10, on the training loss) and epoch cap
GLASS_LR_PATIENCE, GLASS_MIN_LR, GLASS_MAX_EPOCHS = 10, 5e-5, 300


def glass_recipe(name):
    """The recipe's hps for a dataset, from GLASS's config (readout = GLASS's pooling)."""
    cfg = GLASS_CONFIGS[name]
    return {"batch_size": cfg["batch_size"], "lr": cfg["lr"], "lr_on": "train", "lr_factor": cfg["resi"],
            "glass_stop": True, "norm": "graph", "input_norm": "graph", "input_dropout": cfg["dropout"],
            "msg_dropout": cfg["dropout"], "readout": cfg["pool"], "color_mlp_hidden_dims": 0}
CASE_READOUTS = {"pool": ["multistat", "gated_attn"],  # the only winners of the first run
                 "transformer": ["transformer"], "extra_node": ["extra_node"], "hub_graph": ["hub_graph"],
                 "clique_sym": None, "clique_sym_w": None, "no_clique": None,  # None: pool's own best readout
                 "gin_ft": None, "member_copies": None, "edge_label": None, "copies_plus": None,
                 "copies_skip": None, "plain_gnn": None, "plain_gnn_fix": None, "copies_skip_fix": None,
                 "plain_gnn_glass": None, "copies_glass": None, "edge_label_glass": None,
                 "copies_skip_tune": None, "glass_plain": None, "edge_label_recipe": None,
                 "copies_recipe": None}
REAL_ONLY_CASES = ("gin_ft", "plain_gnn", "plain_gnn_fix", "glass_plain", "edge_label_recipe") + GLASS_INPUT_CASES
SYNTHETIC_ONLY_CASES = ("copies_skip_tune", "copies_recipe")
COUNTING_CASES = ("member_copies", "edge_label", "copies_plus", "copies_skip", "copies_skip_fix")  # forced to RGCN
# The case whose best config each case starts from (default pool)
BASE_CASE = {"copies_skip": "copies_plus", "plain_gnn_fix": "plain_gnn", "copies_skip_fix": "copies_skip",
             "plain_gnn_glass": "plain_gnn_fix", "copies_glass": "copies_skip_fix",
             "edge_label_glass": "edge_label", "copies_skip_tune": "copies_skip_fix",
             "edge_label_recipe": "edge_label", "copies_recipe": "copies_skip_tune"}
TRAIN_FIX_CASES = ("plain_gnn_fix", "copies_skip_fix", "copies_skip_tune")
# GLASS's GNN-plain (its appendix A.10): hidden 64, dropout 0.5, fine-tuned pretrained input
PLAIN_GNN_HPS = {"backbone": "rgcn", "gnn_embedding_dim": 64, "gnn_hidden_dim": 64, "gnn_dropout_rate": 0.5,
                 "gnn_mlp_hidden_dims": 64, "mlp_dropout_rate": 0.5, "color_mlp_hidden_dims": 64,
                 "weight_decay": 0.0, "node_input": "gin_ft", "feature_norm": "standardize",
                 "clique_norm": "none", "node_loss_weight": 0.0}
GIN_CAT_LEARNED_DIM = 64  # the learned part of gin_cat, when pool's input had no size of its own
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
                "edge_label": {"lr": [1e-3, 3e-3, 1e-2]},
                # readout and, on the real datasets, node_input are added by small_grid
                "copies_plus": {"lr": [1e-3, 3e-3, 1e-2]},
                "copies_skip": {"lr": [1e-3, 3e-3, 1e-2]},
                # GLASS searched lr 1e-4..5e-3, 1-2 layers, mean/sum/gcn aggregation, mean/sum/max/size pool
                "plain_gnn": {"lr": [1e-3, 5e-3], "k_gnn_layers": [1, 2], "graph_agg": ["mean", "gcn"],
                              "readout": ["mean", "multistat"]},
                # n_batches: train subgraphs per epoch split into this many steps (GLASS: 10-80)
                "plain_gnn_fix": {"norm": ["batch", "layer"], "n_batches": [1, 10]},
                "copies_skip_fix": {"norm": ["batch", "layer"], "n_batches": [1, 10]},
                "plain_gnn_glass": {"lr": [1e-3, 5e-3], "gnn_dropout_rate": [0.2, 0.5], "k_gnn_layers": [1, 2],
                                    "norm": ["batch", "layer"]},
                # hpo_metab takes ~10 min a run here, so this grid stays small
                "copies_glass": {"lr": [1e-3, 3e-3], "gnn_dropout_rate": [0.0, 0.5]},
                "edge_label_glass": {"norm": [None, "batch", "layer"], "lr": [1e-3, 3e-3],
                                     "gnn_dropout_rate": [0.0, 0.5]},
                "copies_skip_tune": {"lr": [1e-3, 3e-3, 1e-2], "k_gnn_layers": [1, 2, 3],
                                     "gnn_dropout_rate": [0.0, 0.2], "norm": ["batch", "layer"]},
                # The recipe grids depend on the dataset's base config and GLASS's; small_grid fills them
                "glass_plain": {}, "edge_label_recipe": {}, "copies_recipe": {}}
POOL_ONLY_HPS = ["attn_pool_hidden_dim", "attn_pool_dropout"]  # gated attention's own settings
# Each case gets its own results folder; pool is the v2 run, so it keeps that folder
CASE_DIRS = {"pool": "gpen_compare_v2", "transformer": "gpen_compare_transformer",
             "extra_node": "gpen_compare_extra_node", "hub_graph": "gpen_compare_hub_graph",
             "clique_sym": "gpen_compare_clique_sym", "clique_sym_w": "gpen_compare_clique_sym_w",
             "no_clique": "gpen_compare_no_clique", "gin_ft": "gpen_compare_gin_ft",
             "member_copies": "gpen_compare_member_copies", "edge_label": "gpen_compare_edge_label",
             "copies_plus": "gpen_compare_copies_plus", "copies_skip": "gpen_compare_copies_skip",
             "plain_gnn": "gpen_compare_plain_gnn", "plain_gnn_fix": "gpen_compare_plain_gnn_fix",
             "copies_skip_fix": "gpen_compare_copies_skip_fix", "plain_gnn_glass": "gpen_compare_plain_gnn_glass",
             "copies_glass": "gpen_compare_copies_glass", "edge_label_glass": "gpen_compare_edge_label_glass",
             "copies_skip_tune": "gpen_compare_copies_skip_tune", "glass_plain": "gpen_compare_glass_plain",
             "edge_label_recipe": "gpen_compare_edge_label_recipe", "copies_recipe": "gpen_compare_copies_recipe"}
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
            if key == "glass" and not os.path.exists(path):
                # em_user's GLASS vectors were never copied over; read them from the GLASS clone
                path = join("external", "GLASS", "Emb", f"{name}_64.pt")
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
    member_copies' relations among the copies, in membership rows (row r is one
    (node, subgraph) pair; rows are grouped by subgraph, in the order of
    `subgraphs`). copy_rows maps the rows to graph ids.
    :return: (inside [2, E_in]: rows of one subgraph whose nodes are graph
              neighbors, clique [2, E_cl]: every pair of rows of one subgraph),
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
    clique = torch.stack([ri, rj])
    return clique[:, is_edge], clique


def copy_rows(member_node, n_nodes, copy_all=False):
    """
    The graph id of each membership row's copy: a new node n_nodes + k for a
    shared node (in two or more subgraphs), the node itself otherwise.
    copy_all gives every row a new node, as in runs 5-10.
    :return: (rep [number of memberships], shared [number of memberships] bool)
    """
    if copy_all:
        shared = torch.ones(member_node.numel(), dtype=torch.bool)
    else:
        shared = torch.bincount(member_node, minlength=n_nodes)[member_node] >= 2
    rep = member_node.clone()
    rep[shared] = n_nodes + torch.arange(int(shared.sum()))
    return rep, shared


def copy_counts(inside, graph_edges, member_node, n_nodes):
    """
    copies_plus' input counts: each copy's graph edges inside its own subgraph,
    and the rest of its node's edges (outside), log1p and standardized.
    :param inside: inside edges in membership rows (build_copy_edges)
    :return: [number of memberships, 2] float tensor
    """
    n_members = member_node.numel()
    in_deg = torch.bincount(torch.cat([inside[0], inside[1]]), minlength=n_members).float()
    deg = torch.bincount(torch.cat([graph_edges[0], graph_edges[1]]), minlength=n_nodes).float()
    x = torch.stack([in_deg.log1p(), (deg[member_node] - in_deg).log1p()], dim=1)
    return (x - x.mean(0)) / x.std(0).clamp(min=1e-6)


def subgraph_counts(inside, graph_edges, member_node, member_sub, n_nodes, n_subs):
    """
    copies_skip's per-subgraph counts: size, graph edges inside, and graph edges
    leaving it (cut_ratio's numerator), log1p and standardized.
    :return: [n_subs, 3] float tensor
    """
    n_members = member_node.numel()
    in_deg = torch.bincount(torch.cat([inside[0], inside[1]]), minlength=n_members).float()
    deg = torch.bincount(torch.cat([graph_edges[0], graph_edges[1]]), minlength=n_nodes).float()
    size = torch.bincount(member_sub, minlength=n_subs).float()
    # Each inside edge is counted at both of its ends
    inside_edges = torch.zeros(n_subs).index_add_(0, member_sub, in_deg) / 2
    outside_edges = torch.zeros(n_subs).index_add_(0, member_sub, deg[member_node] - in_deg)
    x = torch.stack([size, inside_edges, outside_edges], dim=1).log1p()
    return (x - x.mean(0)) / x.std(0).clamp(min=1e-6)


def graph_gcn_weights(graph_edges, n_nodes):
    """plain_gnn's GCN aggregation: 1 / sqrt(d_a d_b) per graph edge, summed. :return: [E]"""
    a, b = graph_edges
    deg = torch.bincount(torch.cat([a, b]), minlength=n_nodes).float()
    return (deg[a] * deg[b]).rsqrt()


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


def split_node_targets(split, y_hot, member_node, member_sub, n_nodes):
    """
    Node-head targets, per split: the mean label of that split's subgraphs that
    contain the node. Soft class distribution (multi-class) or per-class
    frequency (multi-label).
    :return: ({split: [n_nodes, n_classes] targets}, {split: [n_nodes] bool mask})
    """
    node_targets, node_masks = {}, {}
    for s in SPLITS.values():
        sel = split[member_sub] == s
        t = torch.zeros(n_nodes, y_hot.size(1)).index_add_(0, member_node[sel], y_hot[member_sub[sel]])
        cnt = torch.bincount(member_node[sel], minlength=n_nodes)
        node_targets[s] = t / cnt.clamp(min=1).unsqueeze(1).float()
        node_masks[s] = cnt > 0
    return node_targets, node_masks


def glass_split(prep, name, repeat):
    """
    prep on the split GLASS's code scores repeat `repeat` on. Real datasets: the
    shipped split, with val and test swapped when val is the smaller (its loader
    does this; only ppi_bp, 159 vs 160). Synthetic: GLASS ignores the shipped
    split and draws 50% train / 25% val / 25% test per repeat, as the first
    random call after seeding torch with 2^repeat - 1, so this reproduces its
    exact splits.
    :return: a shallow copy of prep with split and the node-head targets replaced
    """
    split = prep["split"].clone()
    if name in REAL:
        if (split == 1).sum() < (split == 2).sum():
            split = torch.where(split == 1, 2, torch.where(split == 2, 1, split))
    else:
        cnt = split.numel()
        base = torch.cat([torch.zeros(cnt - cnt // 2, dtype=torch.long), torch.ones(cnt // 4, dtype=torch.long),
                          2 * torch.ones(cnt // 2 - cnt // 4, dtype=torch.long)])
        torch.manual_seed((1 << repeat) - 1)
        split = base[torch.randperm(cnt)]
    node_targets, node_masks = split_node_targets(split, prep["y_hot"], prep["member_node"], prep["member_sub"],
                                                  prep["n_nodes"])
    return {**prep, "split": split, "node_targets": node_targets, "node_masks": node_masks}


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
    node_targets, node_masks = split_node_targets(split, y_hot, member_node, member_sub, n_nodes)

    same_sub_edges = build_same_subgraph_edges(ds["subgraphs"], n_nodes)
    struct = structural_features(ds["edges"], same_sub_edges, member_node, n_nodes)
    # Supernode of subgraph c is node n_nodes + c, linked to every member
    supernode_edges = torch.stack([member_node, n_nodes + member_sub])

    return {"n_nodes": n_nodes, "n_subs": n_subs, "n_classes": n_classes, "classes": classes,
            "multilabel": multilabel, "y": y, "y_hot": y_hot, "split": split,
            "member_node": member_node, "member_sub": member_sub,
            "node_targets": node_targets, "node_masks": node_masks,
            "graph_edges": ds["edges"], "same_sub_edges": same_sub_edges,
            "supernode_edges": supernode_edges, "member_pairs": None, "hub_edges": {},
            "copy_edges": None, "graph_inside": None, "edge_kinds": {},  # built on first use
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
    if hps.get("coloring"):
        return build_colored_relations(prep, hps)
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
    elif hps.get("graph_agg") == "gcn":
        rel_weights = [graph_gcn_weights(prep["graph_edges"], prep["n_nodes"])]
        sum_relations = [0]
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


def outside_copy_edges(graph_edges, member_node, member_sub, rep, n_nodes):
    """
    The graph edges once the shared nodes are replaced by their copies: edge u-v
    joins every copy of u to every copy of v (a node that is not shared is its
    own single copy), and is kept here unless the two copies are of one subgraph
    (those are the inside edges).
    :return: [2, E_out]
    """
    # One entry per (node, its copy, the copy's subgraph): the membership rows, plus the
    # nodes in no subgraph (subgraph -1)
    plain = (torch.bincount(member_node, minlength=n_nodes) == 0).nonzero(as_tuple=True)[0]
    ent_node = torch.cat([member_node, plain])
    ent_rep = torch.cat([rep, plain])
    ent_sub = torch.cat([member_sub, torch.full((plain.numel(),), -1, dtype=torch.long)])
    order = torch.argsort(ent_node, stable=True)
    ent_node, ent_rep, ent_sub = ent_node[order], ent_rep[order], ent_sub[order]
    cnt = torch.bincount(ent_node, minlength=n_nodes)
    ptr = torch.cumsum(cnt, 0) - cnt
    # Every (entry of u, entry of v) pair of every edge
    u, v = graph_edges
    n_pairs = cnt[u] * cnt[v]
    edge = torch.repeat_interleave(torch.arange(u.numel()), n_pairs)
    k = torch.arange(int(n_pairs.sum())) - torch.repeat_interleave(torch.cumsum(n_pairs, 0) - n_pairs, n_pairs)
    eu = ptr[u[edge]] + k // cnt[v[edge]]
    ev = ptr[v[edge]] + k % cnt[v[edge]]
    inside = (ent_sub[eu] == ent_sub[ev]) & (ent_sub[eu] >= 0)
    return torch.stack([ent_rep[eu[~inside]], ent_rep[ev[~inside]]])


def sibling_copy_edges(member_node, rep, shared):
    """Every pair of copies of one shared node, each unordered pair once. :return: [2, E_sib]"""
    node, copy = member_node[shared], rep[shared]
    order = torch.argsort(node, stable=True)
    node, copy = node[order], copy[order]
    first, second = [], []
    for start, k in zip((torch.cumsum(torch.bincount(node), 0) - torch.bincount(node)).tolist(),
                        torch.bincount(node).tolist()):
        if k >= 2:
            i, j = torch.triu_indices(k, k, offset=1)
            first.append(copy[start + i])
            second.append(copy[start + j])
    if not first:
        return torch.empty(2, 0, dtype=torch.long)
    return torch.stack([torch.cat(first), torch.cat(second)])


def build_copy_relations(prep, hps):
    """
    member_copies' relations. A shared node (in two or more subgraphs) is
    replaced by its copies, one per subgraph, each with all of the node's graph
    edges: outside edges (mean), inside edges to the copies of the same
    subgraph (summed), and sibling edges between the copies of one node (mean);
    with copy_clique also every pair of copies of one subgraph (mean). The
    shared node's own row keeps no edges.
    copy_all (runs 5-10): every member gets a copy next to its node: the graph
    (mean), each node to its copies (mean), inside edges among the copies
    (summed), and the copy clique. Same return as build_relations.
    """
    n_nodes, member_node = prep["n_nodes"], prep["member_node"]
    if prep["copy_edges"] is None:
        prep["copy_edges"] = build_copy_edges(prep["subgraphs"], prep["graph_edges"], n_nodes)
    inside, clique = prep["copy_edges"]
    copy_all = hps.get("copy_all", False)
    rep, shared = copy_rows(member_node, n_nodes, copy_all)
    if copy_all:
        copy_links = torch.stack([member_node[shared], rep[shared]])
        rel_edges, sum_relations = [prep["graph_edges"], copy_links, rep[inside]], [2]
    else:
        if prep.get("outside_copy_edges") is None:
            prep["outside_copy_edges"] = outside_copy_edges(prep["graph_edges"], member_node, prep["member_sub"],
                                                            rep, n_nodes)
        rel_edges, sum_relations = [prep["outside_copy_edges"], rep[inside],
                                    sibling_copy_edges(member_node, rep, shared)], [1]
    if hps["copy_clique"]:
        rel_edges.append(rep[clique])
    rel_types = [torch.full((e.size(1),), r, dtype=torch.long) for r, e in enumerate(rel_edges)]
    return torch.cat(rel_edges, dim=1), torch.cat(rel_types), len(rel_edges), None, sum_relations


# The edge kinds of the copies graph (copy_edge_kinds), then of the graph without copies
# (graph_edge_kinds); "I", "O" and "G" in a coloring stand for both of theirs
EDGE_KINDS = ("Icopy", "Iplain", "Osub", "Onone", "Oin", "Oout", "S", "C")
GRAPH_KINDS = ("Gin", "Gout")
KIND_GROUPS = {"I": ("Icopy", "Iplain"), "O": ("Osub", "Onone"), "G": ("Gin", "Gout")}


def graph_edge_kinds(prep, kinds):
    """
    The dataset's own graph, no copies: Gin the edges whose two ends share a
    subgraph (edge_label's inside edges), Gout the rest. Directed edge lists, as
    in copy_edge_kinds.
    :return: dict kind -> [2, E]
    """
    edges = prep["graph_edges"]
    inside = inside_edge_mask(edges, prep["same_sub_edges"], prep["n_nodes"])
    out = {"Gin": edges[:, inside], "Gout": edges[:, ~inside]}
    return {k: torch.cat([out[k], out[k].flip(0)], dim=1) for k in kinds}


def copy_edge_kinds(prep, kinds):
    """
    The graph once every shared node is replaced by its copies (as in
    build_copy_relations), its edges sorted into kinds, in graph ids (copy_rows):
      Icopy / Iplain: inside edges that touch a copy / that join two nodes that are no copy
      Osub / Onone  : outside edges between members of two subgraphs / with an end in no subgraph
      Oin / Oout    : the outside edges' messages, by whether the receiver is in a subgraph
      S             : sibling edges, between two copies of one node
      C             : every pair of members (nodes or copies) of one subgraph, our clique
    Every kind is a directed edge list, an undirected edge in both directions.
    :param kinds: the kinds to build
    :return: dict kind -> [2, E]
    """
    n_nodes, member_node = prep["n_nodes"], prep["member_node"]
    rep, shared = copy_rows(member_node, n_nodes)
    if prep["copy_edges"] is None:
        prep["copy_edges"] = build_copy_edges(prep["subgraphs"], prep["graph_edges"], n_nodes)
    inside, clique = prep["copy_edges"]

    def both_ways(e):
        return torch.cat([e, e.flip(0)], dim=1)

    out = {}
    if kinds & {"Icopy", "Iplain"}:
        touches_copy = shared[inside[0]] | shared[inside[1]]
        out["Icopy"] = both_ways(rep[inside[:, touches_copy]])
        out["Iplain"] = both_ways(rep[inside[:, ~touches_copy]])
    if kinds & {"Osub", "Onone", "Oin", "Oout"}:
        if prep.get("outside_copy_edges") is None:
            prep["outside_copy_edges"] = outside_copy_edges(prep["graph_edges"], member_node, prep["member_sub"],
                                                            rep, n_nodes)
        outside = prep["outside_copy_edges"]
        is_member = torch.zeros(n_nodes + int(shared.sum()), dtype=torch.bool)
        is_member[rep] = True
        both_members = is_member[outside[0]] & is_member[outside[1]]
        out["Osub"] = both_ways(outside[:, both_members])
        out["Onone"] = both_ways(outside[:, ~both_members])
        messages = both_ways(outside)
        out["Oin"] = messages[:, is_member[messages[1]]]
        out["Oout"] = messages[:, ~is_member[messages[1]]]
    if "S" in kinds:
        out["S"] = both_ways(sibling_copy_edges(member_node, rep, shared))
    if "C" in kinds:
        out["C"] = both_ways(rep[clique])
    return {k: out[k] for k in kinds}


def directed_gcn_weights(edges):
    """GCN's D^-1/2 A D^-1/2 for one directed relation: 1 / sqrt(out-degree(src) in-degree(dst)). :return: [E]"""
    if edges.numel() == 0:
        return torch.empty(0)
    n = int(edges.max()) + 1
    out_deg = torch.bincount(edges[0], minlength=n).float()
    in_deg = torch.bincount(edges[1], minlength=n).float()
    return (out_deg[edges[0]] * in_deg[edges[1]]).rsqrt()


def build_colored_relations(prep, hps):
    """
    run_edge_colors' relations: one relation per color of hps["coloring"], a list
    of colors, each a list of edge kinds (copy_edge_kinds, or graph_edge_kinds for
    the graph without copies, hps["copies"] off); a kind in no color is left out of
    the graph. The edges are directed lists, so the GNN runs directed.
    graph_agg: "mean" within each relation, "sum", or "gcn" (D^-1/2 A D^-1/2
    within each relation, summed). The kinds are cached in prep["edge_kinds"].
    Same return as build_relations.
    """
    colors = [[k for name in color for k in KIND_GROUPS.get(name, (name,))] for color in hps["coloring"]]
    cache = prep["edge_kinds"]
    missing = {k for color in colors for k in color} - set(cache)
    if missing & set(GRAPH_KINDS):
        cache.update(graph_edge_kinds(prep, missing & set(GRAPH_KINDS)))
    if missing - set(GRAPH_KINDS):
        cache.update(copy_edge_kinds(prep, missing - set(GRAPH_KINDS)))
    rel_edges = [torch.cat([cache[k] for k in color], dim=1) for color in colors]
    agg = hps.get("graph_agg", "mean")
    sum_relations = list(range(len(rel_edges))) if agg in ("sum", "gcn") else []
    weights = torch.cat([directed_gcn_weights(e) for e in rel_edges]) if agg == "gcn" else None
    rel_types = [torch.full((e.size(1),), r, dtype=torch.long) for r, e in enumerate(rel_edges)]
    return torch.cat(rel_edges, dim=1), torch.cat(rel_types), len(rel_edges), weights, sum_relations


def node_input_choices(prep):
    """Node inputs available for a dataset: learned and structural always, pretrained if shipped."""
    return ["learned", "struct"] + sorted(prep["features"])


def build_input(prep, hps):
    """
    :return: (x, module, in_dim). x is a fixed tensor, module an nn.Embedding
             whose weight is the input; with both (gin_cat) the input is [x, weight].
    """
    kind, n = hps["node_input"], prep["n_nodes"]
    if kind in ("learned", "gin_cat"):
        emb = nn.Embedding(n, hps["node_input_dim"])
        nn.init.normal_(emb.weight, std=0.1)
        if kind == "learned":
            return None, emb, hps["node_input_dim"]
    if kind == "struct":
        return prep["struct"], None, prep["struct"].size(1)
    # <name>_ft: those pretrained vectors as the starting point of a trainable input
    pretrained = "gin" if kind == "gin_cat" else kind.removesuffix("_ft")
    # Pretrained embeddings are unsupervised, so the norm statistics may use every node
    x = normalize_features(prep["features"][pretrained], torch.ones(n, dtype=torch.bool),
                           hps.get("feature_norm", "standardize"))
    if kind.endswith("_ft"):
        return None, nn.Embedding.from_pretrained(x.clone(), freeze=False), x.size(1)
    if kind == "gin_cat":
        return x, emb, x.size(1) + hps["node_input_dim"]
    return x, None, x.size(1)


class SumReadout(nn.Module):
    """GLASS's sum pooling, or with size_norm its "size" pooling: the sum divided by sqrt(subgraph size)."""

    def __init__(self, dim, size_norm=False):
        super().__init__()
        self.out_dim = dim
        self.size_norm = size_norm

    def forward(self, h, color_indices, n_colors):
        out = torch.zeros(n_colors, h.size(1), device=h.device, dtype=h.dtype).index_add_(0, color_indices, h)
        if self.size_norm:
            out = out / torch.bincount(color_indices, minlength=n_colors).clamp(min=1).sqrt().unsqueeze(1)
        return out


class MaxReadout(nn.Module):
    """GLASS's max pooling: the elementwise max over a subgraph's members."""

    def __init__(self, dim):
        super().__init__()
        self.out_dim = dim

    def forward(self, h, color_indices, n_colors):
        index = color_indices.unsqueeze(1).expand_as(h)
        return torch.zeros(n_colors, h.size(1), device=h.device, dtype=h.dtype).scatter_reduce(
            0, index, h, reduce="amax", include_self=False)


def make_gpen_readout(readout, dim, hps):
    """train09's modules: the transformer is its PMA, the extra node its supernode (no module)."""
    if readout in ("sum", "size"):
        return SumReadout(dim, size_norm=readout == "size")
    if readout == "max":
        return MaxReadout(dim)
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
    # member_copies: rows n_nodes.. are the copies of the shared nodes; copy_rep[r] is the
    # graph row of membership r's copy (the node itself when it is in one subgraph)
    copies = bool(hps.get("copies"))

    feat, feat_module, feat_dim = build_input(prep, hps)
    input_dim = feat_dim  # before the copy flag and counts, which the input norm leaves alone
    rel_edge_index, rel_edge_type, num_relations, rel_edge_weight, sum_relations = build_relations(prep, hps)
    counts = copy_rep = copy_src = replaced = None
    if copies:
        copy_rep, shared = copy_rows(prep["member_node"], n_nodes, hps.get("copy_all", False))
        copy_src = prep["member_node"][shared].to(DEVICE)
        n_rows = n_nodes + int(shared.sum())
        # Without copy_all a shared node is replaced by its copies; the node head reads their mean
        replaced = None
        if not hps.get("copy_all", False):
            replaced = (torch.bincount(prep["member_node"], minlength=n_nodes) >= 2).to(DEVICE)
        feat_dim += 1  # the copy flag: 1 on the copies, 0 elsewhere
        if hps.get("copy_counts"):
            # copies_plus: each copy's inside and outside edge counts (zeros on nodes that are no copy)
            row_counts = copy_counts(prep["copy_edges"][0], prep["graph_edges"], prep["member_node"], n_nodes)
            counts = torch.zeros(n_rows, 2).index_copy_(0, copy_rep, row_counts).to(DEVICE)
            feat_dim += 2
        copy_rep = copy_rep.to(DEVICE)
    sub_counts = None
    if hps.get("count_skip"):
        sub_counts = subgraph_counts(prep["copy_edges"][0], prep["graph_edges"], prep["member_node"],
                                     prep["member_sub"], n_nodes, n_subs).to(DEVICE)
    node_loss_weight = hps.get("node_loss_weight", 1.0)

    pair_edge_index = None
    if attention_pool and hps["pma_sabs"] > 0:
        if prep["member_pairs"] is None:
            prep["member_pairs"] = build_member_pairs(prep["member_sub"])
        pair_edge_index = prep["member_pairs"].to(DEVICE)

    # --- Modules
    modules = {}
    if feat_module is not None:
        modules["node_input"] = feat_module
    if hps.get("input_norm") == "graph":
        modules["input_norm"] = GraphNorm(input_dim)
    # Colored relations come as directed edge lists (build_colored_relations)
    directed = bool(hps.get("coloring"))
    if backbone == "rgcn":
        modules["gnn"] = RGCN(in_dim=feat_dim, hidden_dim=hps["gnn_hidden_dim"],
                              out_dim=hps["gnn_embedding_dim"], num_relations=num_relations,
                              num_layers=hps["k_gnn_layers"], dropout=hps["gnn_dropout_rate"], directed=directed,
                              sum_relations=sum_relations, norm=hps.get("norm"), jk=hps.get("jk", False),
                              msg_dropout=hps.get("msg_dropout", 0.0))
    else:
        # RGAT has no sum mode: its softmax renormalizes, so the weights only bias the attention
        modules["gnn"] = RGAT(in_dim=feat_dim, hidden_dim=hps["gnn_hidden_dim"],
                              out_dim=hps["gnn_embedding_dim"], num_relations=num_relations,
                              num_layers=hps["k_gnn_layers"], num_heads=hps["num_heads"],
                              dropout=hps["gnn_dropout_rate"], attn_dropout=hps["gnn_dropout_rate"],
                              directed=directed, norm=hps.get("norm"), jk=hps.get("jk", False))
    gnn_out = getattr(modules["gnn"], "out_dim", hps["gnn_embedding_dim"])  # wider with jumping knowledge
    modules["node_mlp"] = MLP(input_dim=gnn_out, hidden_dims=[hps["gnn_mlp_hidden_dims"]],
                              output_dim=n_classes, dropout_rate=hps["mlp_dropout_rate"])
    pooling = make_gpen_readout(readout, gnn_out, hps)
    if pooling is not None:
        modules["readout"] = pooling
    if readout != "node_score_mean":
        color_in = gnn_out if supernode else pooling.out_dim
        if sub_counts is not None:
            color_in += sub_counts.size(1)
        # color_mlp_hidden_dims 0: a single linear layer, GLASS's head
        hidden = [hps["color_mlp_hidden_dims"]] if hps["color_mlp_hidden_dims"] else []
        modules["color_head"] = MLP(input_dim=color_in, hidden_dims=hidden,
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
    def color_loss(out, s, idx=None):
        idx = sub_idx[s] if idx is None else idx
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
    # lr_on "train": GLASS's decay, on the training loss with its factor and patience
    lr_on_train = hps.get("lr_on") == "train"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=hps["lr_factor"] if lr_on_train else LR_FACTOR,
        patience=GLASS_LR_PATIENCE if lr_on_train else LR_PATIENCE, min_lr=GLASS_MIN_LR if lr_on_train else MIN_LR)

    # batch_size (GLASS's) sets the mini-batches; otherwise n_batches does
    n_train = len(sub_idx[0])
    n_batches = -(-n_train // hps["batch_size"]) if hps.get("batch_size") else hps.get("n_batches", 1)
    min_epochs, patience = MIN_EPOCHS, PATIENCE
    # GLASS's early stopping: selection starts after 100 / num_div epochs and stops after as
    # many without a better validation score; num_div = test size / batch size, over 5 on the
    # synthetic datasets. Only with GLASS's batch size: one full batch would make it wait
    # past the epoch cap
    glass_stop = bool(hps.get("glass_stop") and hps.get("batch_size"))
    if glass_stop:
        num_div = len(sub_idx[2]) / hps["batch_size"]
        if prep.get("name") not in REAL:
            num_div /= 5
        min_epochs = patience = 100 / num_div
        epochs = min(epochs, GLASS_MAX_EPOCHS)

    def node_x():
        """The GNN input, with the extra node rows appended when the readout needs them."""
        if feat_module is None:
            x = feat
        elif feat is None:
            x = modules["node_input"].weight
        else:
            x = torch.cat([feat, modules["node_input"].weight], dim=1)  # gin_cat
        if "input_norm" in modules:
            x = modules["input_norm"](x)
        x = F.dropout(x, p=hps.get("input_dropout", 0.0), training=modules["gnn"].training)
        if copies:
            # A copy starts from its node's input; the flag tells the two apart
            x = torch.cat([x, x[copy_src]], dim=0)
            flag = torch.zeros(x.size(0), 1, device=x.device)
            flag[n_nodes:] = 1.0
            return torch.cat([x, flag] + ([counts] if counts is not None else []), dim=1)
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
        if copies and replaced is not None:
            h_nodes = torch.where(replaced.unsqueeze(1), scatter_mean(h[n_nodes:], copy_src, n_nodes), h_nodes)
        node_logits = modules["node_mlp"](h_nodes)

        if copies:
            # Pool the copies, so each subgraph reads only its own view of a shared node
            h_copies = h[copy_rep]  # one row per membership, so the member pairs index it directly
            if attention_pool:
                pooled = modules["readout"](h_copies, member_sub, n_subs, pair_edge_index)
            else:
                pooled = modules["readout"](h_copies, member_sub, n_subs)
            if sub_counts is not None:
                pooled = torch.cat([pooled, sub_counts], dim=1)
            color_out = modules["color_head"](pooled)
        elif readout == "node_score_mean":
            node_probs = torch.sigmoid(node_logits) if multilabel else torch.softmax(node_logits, dim=1)
            color_out = scatter_mean(node_probs[member_node], member_sub, n_subs)
        elif supernode:
            color_out = modules["color_head"](h[n_nodes:])
        else:
            if attention_pool:
                pooled = modules["readout"](h_nodes[member_node], member_sub, n_subs, pair_edge_index)
            else:
                pooled = modules["readout"](h_nodes[member_node], member_sub, n_subs)
            if sub_counts is not None:
                pooled = torch.cat([pooled, sub_counts], dim=1)
            color_out = modules["color_head"](pooled)
        return node_logits, color_out

    # A probability of 0.5 is a logit of 0
    threshold = 0.5 if readout == "node_score_mean" else 0.0

    def val_f1(color_out, s=1):
        """Micro-F1 on the validation subgraphs (or split s), on the GPU (accuracy when single-label)."""
        out, target = color_out[sub_idx[s]], y[sub_idx[s]]
        if not multilabel:
            return (out.argmax(1) == target).float().mean().item()
        pred, true = out > threshold, target > 0.5
        tp = (pred & true).sum()
        return (2 * tp / (2 * tp + (pred & ~true).sum() + (~pred & true).sum()).clamp(min=1)).item()

    # --- Training loop. Stop after PATIENCE epochs without a better validation
    # micro-F1 (ties go to the lower validation loss), never before MIN_EPOCHS.
    # Validation is scored in eval mode, so dropout does not blur the selection.
    # With glass_stop, epochs before min_epochs are not candidates (GLASS does not score them).
    best_score, unimproved_epochs, best_states = (-1.0, -torch.inf), 0, None
    # glass_select: also record the test score GLASS's selection would report (GLASSTest.py), for
    # comparison only - it reads test, so it never picks anything. Among the epochs tied on the best
    # validation micro-F1, the best test. Training, stopping and test_f1 are unchanged; since our
    # stopping counts tied epochs as unimproved (GLASS's does not), it can only under-count GLASS's.
    glass_select = bool(hps.get("glass_select"))
    glass_val, glass_test, test_now = -1.0, -1.0, None
    epoch = 0
    for epoch in range(epochs):
        for m in modules.values():
            m.train()
        # One step per mini-batch of train subgraphs; the node loss is split evenly over the steps
        train_losses = []
        for batch in torch.randperm(n_train, device=DEVICE).chunk(n_batches):
            node_logits, color_out = forward()
            loss = (node_loss_weight / n_batches * node_loss(node_logits, 0)
                    + color_loss(color_out, 0, sub_idx[0][batch]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        for m in modules.values():
            m.eval()
        with torch.no_grad():
            node_logits, color_out = forward()
            val_loss = (node_loss_weight * node_loss(node_logits, 1) + color_loss(color_out, 1)).item()
            score = (val_f1(color_out), -val_loss)
            if glass_select:
                test_now = val_f1(color_out, 2)
        scheduler.step(float(np.mean(train_losses)) if lr_on_train else val_loss)

        if glass_stop and epoch < min_epochs:
            continue
        if glass_select:
            if score[0] > glass_val + 1e-5:
                glass_val, glass_test = score[0], test_now
            elif score[0] >= glass_val - 1e-5:
                glass_test = max(glass_test, test_now)
        if score > best_score:
            best_score = score
            unimproved_epochs = 0
            best_states = {name_: {k: v.detach().clone() for k, v in m.state_dict().items()}
                           for name_, m in modules.items()}
        else:
            unimproved_epochs += 1
        if epoch + 1 >= min_epochs and unimproved_epochs >= patience:
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
    if glass_select:
        out["test_f1_glass"] = glass_test
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
    # glass_select runs carry one more column, at the end; other CSVs are unchanged
    fields = TRIAL_FIELDS + [k for k in ("test_f1_glass",) if k in row]
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def make_row(name, hps, trial, seed, res, seconds):
    row = {"dataset": name, "backbone": hps["backbone"], "readout": hps["readout"],
           "trial": trial, "seed": seed,
           "train_f1": round(res["train_f1"], 4), "val_f1": round(res["val_f1"], 4),
           "test_f1": round(res["test_f1"], 4), "epochs": res["epochs"],
           "seconds": round(seconds, 1), "hps": json.dumps(hps)}
    if hps.get("glass_select"):
        row["test_f1_glass"] = round(res.get("test_f1_glass", -1.0), 4)
    return row


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
    if case == "glass_plain":
        # Built from GLASS's config alone; GLASS's head is a single linear layer and it has no node head
        cfg = GLASS_CONFIGS[name]
        return {"backbone": "rgcn", "gnn_embedding_dim": GLASS_HIDDEN, "gnn_hidden_dim": GLASS_HIDDEN,
                "k_gnn_layers": cfg["conv_layer"], "gnn_dropout_rate": cfg["dropout"], "gnn_mlp_hidden_dims": 64,
                "mlp_dropout_rate": 0.0, "weight_decay": 0.0, "node_input": "glass_ft", "feature_norm": "none",
                "clique_norm": "none", "node_loss_weight": 0.0, "graph_agg": cfg["aggr"], "jk": True,
                **glass_recipe(name)}
    path = join(case_dir(BASE_CASE.get(case, "pool")), name, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        hps = json.loads(json.load(f)["hps"])
    if case == "edge_label_recipe":
        # The grid sets the readout (own or GLASS's pooling) and the input (own or GLASS's vectors)
        return {**hps, "backbone": "rgcn", "graph_split": True, **glass_recipe(name), "readout": hps["readout"]}
    if case == "copies_recipe":
        # The grid sets the batch size and the readout; the rest of the recipe is fixed
        recipe = glass_recipe(name)
        fixed = ("lr_on", "lr_factor", "glass_stop", "norm", "input_norm", "input_dropout", "msg_dropout")
        return {**hps, **{k: recipe[k] for k in fixed}}
    if case == "copies_skip":
        return {**hps, "count_skip": True}
    if case in TRAIN_FIX_CASES:
        return hps  # the grid adds the norm and the mini-batches
    if case in GLASS_INPUT_CASES:
        return {**hps, "node_input": "glass_ft", "feature_norm": "standardize"}
    if case == "plain_gnn":
        return dict(PLAIN_GNN_HPS)  # the grid adds lr, layers, aggregation and readout
    if case in CLIQUE_NORMS:
        return {**hps, "clique_norm": CLIQUE_NORMS[case]}
    if case == "gin_ft":
        hps = {k: v for k, v in hps.items() if k != "node_input_dim"}
        return {**hps, "node_input": "gin_ft", "feature_norm": "standardize"}
    if case == "member_copies":
        return {**hps, "backbone": "rgcn", "copies": True}
    if case == "edge_label":
        return {**hps, "backbone": "rgcn", "graph_split": True}
    if case == "copies_plus":
        # The transformer gets one self-attention block, run 3's most common winner
        return {"node_input_dim": GIN_CAT_LEARNED_DIM, **hps, "backbone": "rgcn", "copies": True,
                "copy_clique": False, "copy_counts": True, "pma_heads": TRANSFORMER_HEADS, "pma_sabs": 1,
                "attn_pool_hidden_dim": hps.get("attn_pool_hidden_dim", 32),
                "attn_pool_dropout": hps.get("attn_pool_dropout", 0.0)}
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
    if case == "copies_plus":
        grid["readout"] = [base_hps["readout"] if base_hps else "pool", "transformer"]
        if name in REAL:
            grid["node_input"] = [base_hps["node_input"] if base_hps else "pool", "gin_cat"]
    if case in ("edge_label_recipe", "copies_recipe"):
        # Each axis: the base config's own value, then GLASS's (an axis where they agree has one value)
        recipe = glass_recipe(name)
        own = base_hps or {}
        axes = {"readout": recipe["readout"]}
        if case == "edge_label_recipe":
            axes["node_input"] = "glass_ft"
        else:
            # batch_size None: the base config's own n_batches. GLASS's tiny synthetic batches make
            # a coreness run ~10 min, so lr and the head stay the base config's
            axes["batch_size"] = recipe["batch_size"]
        for key, glass_value in axes.items():
            own_value = own.get(key, "own") if key != "batch_size" else None
            grid[key] = [own_value] if own_value == glass_value else [own_value, glass_value]
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


def run_seeds(name, prep, seeds, out_dir, epochs, hps=None, prep_of=None):
    """
    Retrain a config over `seeds` seeds and write summary.json: the given hps, or
    the study's best. prep_of(seed), when given, is the prep each seed trains on
    (GLASS's per-seed splits). Each seed's test predictions go to test_predictions.csv
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
        res = train(prep_of(seed) if prep_of else prep, hps, epochs=epochs, seed=seed)
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
            int(np.prod([len(v) for v in small_grid(case, name, pool_best_hps(name, case)).values()]))
        return s["n_seeds"] >= args.seeds and s["n_trials"] >= trials

    # Each job is its own process, so all of its memory (RAM and GPU) is
    # returned to the system the moment it ends, before the next one starts.
    jobs = [(case, name) for name in HEAVIEST_FIRST if name not in args.exclude for case in args.cases
            if not (case in REAL_ONLY_CASES and name not in REAL)
            and not (case in SYNTHETIC_ONLY_CASES and name not in SYNTHETIC)]
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
    prep["name"] = args.dataset
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
        if case in SYNTHETIC_ONLY_CASES and args.dataset not in SYNTHETIC:
            print(f"[{args.dataset}] {case} is for the synthetic datasets only - skipping")
            continue
        if case != "pool":
            # Pool's best config with this case's readout, then the small grid on top of it
            hps = pool_best_hps(args.dataset, case)
            if hps is None:
                print(f"[{args.dataset}] no pool result to take the config from - skipping {case}")
                continue
            os.makedirs(out_dir, exist_ok=True)
            seeds, epochs = (2, 15) if args.smoke else (args.seeds, args.epochs)
            case_prep, prep_of = prep, None
            if case in GLASS_EVAL_CASES:
                # GLASS's splits: the grid uses repeat 0's, each seed its own repeat's
                case_prep = glass_split(prep, args.dataset, 0)
                if args.dataset in SYNTHETIC:
                    prep_of = lambda seed: glass_split(prep, args.dataset, seed)  # noqa: E731
            if not args.seeds_only:
                hps = run_small_search(args.dataset, case_prep, out_dir, epochs, hps,
                                       small_grid(case, args.dataset, hps))
            else:
                hps = best_hps_of(out_dir)
            run_seeds(args.dataset, case_prep, seeds, out_dir, epochs, hps, prep_of=prep_of)
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
