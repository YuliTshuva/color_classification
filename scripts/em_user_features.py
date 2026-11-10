"""
Yuli Tshuva.
em_user_features.py - 15 characteristics of em_user's subgraphs (one per user, nodes are
workouts, label: female = 1, male = 0), compared between positives and negatives.

Writes results/em_user_analysis/features.csv (one row per user) and summary.csv (per
feature: pos / neg mean and median, Mann-Whitney p, single-feature AUC).

Usage (laptop, conda env color-classification):
  python scripts/em_user_features.py
"""

# Imports
from os.path import join
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import networkx as nx
import torch

DATA = join("data", "processed", "em_user")
GLASS_EMB = join("external", "GLASS", "Emb", "em_user_64.pt")
OUT = join("results", "em_user_analysis")


def load():
    edges = np.load(join(DATA, "edges.npy"))
    n = len(pd.read_parquet(join(DATA, "nodes.parquet")))
    A = sp.coo_matrix((np.ones(len(edges), dtype=np.float32), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.float32).tocsr()
    ents = pd.read_csv(join(DATA, "entities.csv")).set_index("entity_id").sort_index()
    mem = pd.read_parquet(join(DATA, "memberships.parquet"))
    return A, edges, ents, mem, n


def main():
    os.makedirs(OUT, exist_ok=True)
    A, edges, ents, mem, n = load()
    deg = np.asarray(A.sum(1)).ravel()
    y = ents["label"].to_numpy()
    E = len(ents)
    M = sp.csr_matrix((np.ones(len(mem), dtype=np.float32), (mem["entity_id"], mem["node_id"])), shape=(E, n))
    M.data[:] = 1  # a node listed twice in one user counts once
    n_users_of = np.asarray(M.sum(0)).ravel()  # users per node

    print("k-core numbers (networkx, ~minutes)...", flush=True)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(map(tuple, edges))
    core = np.array([c for _, c in sorted(nx.core_number(G).items())])
    del G

    # Edge counts between users: (M A M^T)[i, j] = edges from user i's nodes to user j's nodes
    MA = (M @ A).tocsr()
    UU = (MA @ M.T).toarray()
    np.fill_diagonal(UU, 0)

    rows = []
    for e in range(E):
        S = M[e].indices
        s = len(S)
        sub = A[S][:, S]
        e_in = sub.nnz / 2
        n_comp, comp_lab = connected_components(sub, directed=False)
        d_in = np.asarray(sub.sum(1)).ravel()
        # Local clustering inside the subgraph: triangles(v) / (d(d-1)/2)
        tri = np.asarray((sub @ sub).multiply(sub).sum(1)).ravel() / 2
        pairs = d_in * (d_in - 1) / 2
        clust = np.where(pairs > 0, tri / np.maximum(pairs, 1), 0.0).mean()
        boundary = deg[S].sum() - 2 * e_in
        # Nodes shared with other users, and the gender of those other users
        shared = n_users_of[S] > 1
        others = M[:, S].toarray()  # users x my nodes
        others[e] = 0
        co_same = others[y == y[e]].sum()
        co_other = others[y != y[e]].sum()
        ed_same = UU[e, y == y[e]].sum()
        ed_other = UU[e, y != y[e]].sum()
        rows.append({
            "entity_id": e, "label": y[e], "split": ents.loc[e, "split"],
            "size": s,
            "internal_edges": e_in,
            "density": 2 * e_in / (s * (s - 1)) if s > 1 else 0.0,
            "n_components": n_comp,
            "largest_comp_frac": np.bincount(comp_lab).max() / s,
            "mean_internal_degree": d_in.mean(),
            "mean_global_degree": deg[S].mean(),
            "cut_ratio": boundary / (s * (n - s)),  # GLASS's definition
            "frac_edges_leaving": boundary / deg[S].sum(),
            "clustering_inside": clust,
            "mean_coreness": core[S].mean(),
            "frac_nodes_shared": shared.mean(),
            "shared_same_gender_frac": co_same / (co_same + co_other) if co_same + co_other > 0 else np.nan,
            "edges_same_gender_frac": ed_same / (ed_same + ed_other) if ed_same + ed_other > 0 else np.nan,
        })
    df = pd.DataFrame(rows)
    df.to_csv(join(OUT, "features.csv"), index=False)

    feats = [c for c in df.columns if c not in ("entity_id", "label", "split")]
    summ = []
    for f in feats:
        d = df[["label", f]].dropna()
        p, q = d[d.label == 1][f], d[d.label == 0][f]
        summ.append({"feature": f, "pos_mean": p.mean(), "neg_mean": q.mean(), "pos_median": p.median(),
                     "neg_median": q.median(), "mw_p": mannwhitneyu(p, q).pvalue,
                     "auc": roc_auc_score(d.label, d[f])})
    summ = pd.DataFrame(summ)
    summ.to_csv(join(OUT, "summary.csv"), index=False)

    # Feature 14: how separable the users are by their mean GLASS vector (5-fold CV AUC),
    # next to the same test on the structural features above
    emb = torch.load(GLASS_EMB, map_location="cpu").detach().numpy()
    X_emb = np.vstack([emb[M[e].indices].mean(0) for e in range(E)])
    pos_c, neg_c = X_emb[y == 1].mean(0), X_emb[y == 0].mean(0)
    cos = pos_c @ neg_c / np.linalg.norm(pos_c) / np.linalg.norm(neg_c)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
    auc_emb = cross_val_score(clf, X_emb, y, cv=cv, scoring="roc_auc")
    # The two *_same_gender_frac features use the other users' labels and carry an artifact (a
    # user is left out of its own class, so its "same" baseline is lower), so they stay out here
    X_str = df[[f for f in feats if "same_gender" not in f]].to_numpy()
    auc_str = cross_val_score(clf, X_str, y, cv=cv, scoring="roc_auc")

    # Feature 15: label balance per split
    bal = df.groupby("split")["label"].agg(["count", "mean"]).rename(columns={"mean": "frac_pos"})
    pos_rate = y.mean()

    pd.set_option("display.width", 200)
    print(summ.round(4).to_string(index=False))
    print(f"\nshared-node / edge baseline (frac of OTHER users with same gender): "
          f"pos {(y.sum() - 1) / (E - 1):.3f}, neg {(E - y.sum() - 1) / (E - 1):.3f}")
    print(f"\nGLASS mean vectors: cosine(pos centroid, neg centroid) = {cos:.4f}; "
          f"5-fold CV AUC = {auc_emb.mean():.3f} +- {auc_emb.std():.3f}")
    print(f"structural features above: 5-fold CV AUC = {auc_str.mean():.3f} +- {auc_str.std():.3f}")
    print(f"\nlabel balance (overall frac pos {pos_rate:.3f}):\n{bal.round(3)}")
    with open(join(OUT, "dataset_level.txt"), "w") as fh:
        fh.write(f"glass_centroid_cosine {cos:.4f}\nglass_cv_auc {auc_emb.mean():.4f} {auc_emb.std():.4f}\n"
                 f"structural_cv_auc {auc_str.mean():.4f} {auc_str.std():.4f}\n{bal.round(3).to_string()}\n")


if __name__ == "__main__":
    main()
