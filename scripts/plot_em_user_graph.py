"""
Yuli Tshuva.
plot_em_user_graph.py - the whole em_user graph (57,333 workouts, 4.57M edges), nodes
colored by the users (subgraphs) they belong to: green = only female users (positive),
red = only male users (negative), gold = both, light gray = no user.

Layout: 32 spectral coordinates of the normalized adjacency, then t-SNE to 2D (no igraph
on the laptop; a force layout of 4.5M edges in networkx does not finish). Cached in
results/em_user_analysis/layout.npy.

One panel: every node plus a 1% random sample of the edges.

Usage (laptop, conda env color-classification):
  python scripts/plot_em_user_graph.py
"""

# Imports
from os.path import join, exists
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

DATA = join("data", "processed", "em_user")
OUT = join("results", "em_user_analysis")
FIG = join("plots", "em_user_graph.png")
EDGE_SAMPLE = 0.01
GREEN, RED, GOLD, GRAY = "#1a9850", "#d73027", "#f1a20b", "#c8c8c8"


def layout(A):
    path = join(OUT, "layout.npy")
    if exists(path):
        return np.load(path)
    print("spectral embedding...", flush=True)
    d = np.asarray(A.sum(1)).ravel()
    Dm = sp.diags(1 / np.sqrt(d))
    vals, vecs = eigsh(Dm @ A @ Dm, k=33, which="LA")
    X = vecs[:, np.argsort(-vals)[1:]]  # drop the trivial top vector
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    print("t-SNE (~10 min)...", flush=True)
    pos = TSNE(2, init="pca", perplexity=50, random_state=0, n_jobs=-1).fit_transform(X)
    np.save(path, pos)
    return pos


def main():
    os.makedirs(OUT, exist_ok=True)
    edges = np.load(join(DATA, "edges.npy"))
    n = len(pd.read_parquet(join(DATA, "nodes.parquet")))
    A = sp.coo_matrix((np.ones(len(edges), dtype=np.float64), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.float64).tocsr()
    pos = layout(A)

    ents = pd.read_csv(join(DATA, "entities.csv")).set_index("entity_id")
    mem = pd.read_parquet(join(DATA, "memberships.parquet")).drop_duplicates()
    mem["label"] = ents.loc[mem["entity_id"], "label"].to_numpy()
    in_pos = np.zeros(n, bool)
    in_neg = np.zeros(n, bool)
    in_pos[mem.loc[mem.label == 1, "node_id"]] = True
    in_neg[mem.loc[mem.label == 0, "node_id"]] = True
    only_pos, only_neg, both = in_pos & ~in_neg, in_neg & ~in_pos, in_pos & in_neg
    none = ~(in_pos | in_neg)
    n_pos_users, n_neg_users = int((ents.label == 1).sum()), int((ents.label == 0).sum())

    rng = np.random.default_rng(0)
    samp = edges[rng.random(len(edges)) < EDGE_SAMPLE]

    fig, ax = plt.subplots(figsize=(12, 13), facecolor="white")
    ax.add_collection(LineCollection(pos[samp], colors="#7f7f7f", linewidths=0.1, alpha=0.025))
    ax.scatter(*pos[none].T, s=1.5, c=GRAY, alpha=0.5, linewidths=0)
    # Draw the smaller class last so it is not hidden
    for mask, c in sorted([(only_neg, RED), (only_pos, GREEN), (both, GOLD)], key=lambda t: -t[0].sum()):
        ax.scatter(*pos[mask].T, s=7, c=c, alpha=0.9, linewidths=0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"em_user: all {n:,} workouts, {EDGE_SAMPLE:.0%} of {len(edges):,} edges", fontsize=16)
    handles = [Line2D([], [], ls="", marker="o", ms=9, color=c, label=l) for c, l in [
        (GREEN, f"female users only ({only_pos.sum():,} nodes, {n_pos_users} users)"),
        (RED, f"male users only ({only_neg.sum():,} nodes, {n_neg_users} users)"),
        (GOLD, f"both ({both.sum():,} nodes)"),
        (GRAY, f"no user ({none.sum():,} nodes)")]]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=13, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    os.makedirs("plots", exist_ok=True)
    fig.savefig(FIG, dpi=400)
    fig.savefig(FIG.replace(".png", ".pdf"))
    print("saved", FIG)


if __name__ == "__main__":
    main()
