"""
The statistics of report 06 (size and how subgraphs sit in the graph) for the
SubGNN datasets: the four real ones from data/raw/subgnn and the four synthetic
ones from data/raw/subgnn_synthetic (the tmp.npy files GLASS published).

Usage:
  python scripts/subgnn_table_stats.py ppi_bp density cut_ratio coreness component
"""

import sys
from os.path import join

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from config import RAW_DIR

SYNTHETIC = ["density", "cut_ratio", "coreness", "component"]


def load(name):
    """Return (edges as an (m, 2) array, list of node arrays, list of label sets)."""
    if name in SYNTHETIC:
        obj = np.load(join(RAW_DIR, "subgnn_synthetic", name, "tmp.npy"), allow_pickle=True).item()
        g = obj["G"]
        nodes = sorted(g.nodes)
        idx = {v: i for i, v in enumerate(nodes)}
        edges = np.array([[idx[a], idx[b]] for a, b in g.edges], dtype=np.int64)
        subs = [np.array([idx[v] for v in s], dtype=np.int64) for s in obj["subG"]]
        labels = [{str(l)} for l in obj["subGLabel"]]
        n = len(nodes)
    else:
        d = join(RAW_DIR, "subgnn", name)
        edges = pd.read_csv(join(d, "edge_list.txt"), sep=" ", header=None, usecols=[0, 1]).to_numpy(np.int64)
        sub = pd.read_csv(join(d, "subgraphs.pth"), sep="\t", header=None, names=["nodes", "label", "split"])
        subs = [np.array([int(x) for x in s.split("-")], dtype=np.int64) for s in sub["nodes"]]
        labels = [set(str(l).split("-")) for l in sub["label"]]
        n = int(max(edges.max(), max(s.max() for s in subs))) + 1
    edges = edges[edges[:, 0] != edges[:, 1]]
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    return n, edges, [np.unique(s) for s in subs], labels


def stats(name):
    n, edges, subs, labels = load(name)
    m = len(edges)
    adj = sp.coo_matrix((np.ones(m), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    adj = adj + adj.T
    sizes = np.array([len(s) for s in subs])

    count = np.zeros(n, dtype=np.int64)
    for s in subs:
        count[s] += 1
    covered = count > 0

    pieces = np.array([connected_components(adj[s][:, s], directed=False)[0] for s in subs])

    # An edge is inside a subgraph when both ends are in the same subgraph
    keys = []
    for s in subs:
        t = sp.triu(adj[s][:, s], k=1).tocoo()
        keys.append(s[t.row] * n + s[t.col])
    inside = len(np.unique(np.concatenate(keys))) if keys else 0

    row = {"dataset": name, "nodes": n, "edges": m, "avg_degree": 2 * m / n,
           "subgraphs": len(subs), "avg_size": sizes.mean(), "median_size": np.median(sizes),
           "min_size": sizes.min(), "max_size": sizes.max(),
           "classes": len(set().union(*labels)),
           "multi_label": any(len(l) > 1 for l in labels),
           "no_subgraph_pct": 100 * (~covered).mean(),
           "two_plus_pct": 100 * (count[covered] >= 2).mean(),
           "pieces": pieces.mean(), "one_piece_pct": 100 * (pieces == 1).mean(),
           "inside_edges_pct": 100 * inside / m}

    # Same-label neighbours: edges between nodes of two different subgraphs,
    # each node in exactly one subgraph; chance from the labels of those nodes
    if not row["multi_label"]:
        owner = np.full(n, -1)
        for i, s in enumerate(subs):
            owner[s[count[s] == 1]] = i
        lab = np.array([next(iter(l)) for l in labels])
        a, b = owner[edges[:, 0]], owner[edges[:, 1]]
        keep = (a >= 0) & (b >= 0) & (a != b)
        a, b = a[keep], b[keep]
        row["cross_edges"] = int(keep.sum())
        row["same_label_pct"] = 100 * (lab[a] == lab[b]).mean() if keep.any() else np.nan
        ends = pd.Series(np.concatenate([lab[a], lab[b]])).value_counts(normalize=True)
        row["chance_endpoints_pct"] = 100 * (ends ** 2).sum()
        nodes_lab = pd.Series(lab[owner[owner >= 0]]).value_counts(normalize=True)
        row["chance_nodes_pct"] = 100 * (nodes_lab ** 2).sum()

    by_label = pd.DataFrame({"label": ["-".join(sorted(l)) for l in labels], "size": sizes,
                             "pieces": pieces})
    return row, by_label.groupby("label").agg(count=("size", "size"), size=("size", "mean"),
                                              pieces=("pieces", "mean"))


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    for name in sys.argv[1:]:
        row, by_label = stats(name)
        print(pd.Series(row).to_string())
        print(by_label.round(2).to_string(), "\n")
