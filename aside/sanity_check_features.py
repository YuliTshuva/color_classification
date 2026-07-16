"""
Throwaway diagnostic: does the raw 64-d episode embedding carry label signal
that generalizes to held-out patients (colors)?

Replays the valid-stays filter from tamar_data_to_graph.py so node order matches
nodes.txt, then runs logistic regression with the SAME train/val/test color split
logic as train06.train().
"""
import os
import numpy as np
import pandas as pd
from os.path import join
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

BASE = join(os.path.dirname(__file__), "..", "data", "tamars_datasets")
DISEASES = sorted(os.listdir(BASE))


def build(disease):
    folder = join(BASE, disease)
    train_emb = np.load(join(folder, "train_embeddings.npy"))
    test_emb = np.load(join(folder, "test_embeddings.npy"))
    train_df = pd.read_csv(join(folder, "data", "train", "listfile.csv"))
    test_df = pd.read_csv(join(folder, "data", "test", "listfile.csv"))

    emb = np.vstack([train_emb, test_emb])
    df = pd.concat([train_df, test_df], ignore_index=True)
    node_split = np.array([0] * len(train_emb) + [1] * len(test_emb))

    # valid-stays filter (patients with both episode1 and episode2)
    valid_stays = [el.split("_")[0] for el in df["stay"] if "episode2" in el]
    valid_stays = [s for s in valid_stays if f"{s}_episode1_timeseries.csv" in df["stay"].values]
    df["stay_id"] = df["stay"].apply(lambda x: x.split("_")[0])

    keep = df.index[df["stay_id"].isin(valid_stays)]
    emb = emb[keep]
    node_split = node_split[keep]
    df = df[df["stay_id"].isin(valid_stays)].reset_index(drop=True)

    # per-patient label (max over episodes)
    stay_labels = df.groupby("stay_id")["y_true"].max().to_dict()
    df["label"] = df["stay_id"].apply(lambda x: stay_labels[x])

    patient_ids = df["stay_id"].astype(str)
    unique_patients = {p: i for i, p in enumerate(patient_ids.unique())}
    color_indices = np.array([unique_patients[p] for p in patient_ids])
    n_colors = len(unique_patients)
    node_labels = df["label"].values.astype(int)
    # color label = label of any node with that color
    color_labels = np.zeros(n_colors, dtype=int)
    color_labels[color_indices] = node_labels
    return emb, color_indices, node_labels, color_labels, node_split, n_colors


def split_colors(n_colors, test_colors):
    all_colors = np.arange(n_colors)
    tv = all_colors[~np.isin(all_colors, test_colors)]
    n_val = max(1, int(len(tv) * 0.2))
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(tv))
    return tv[perm[n_val:]], tv[perm[:n_val]]  # train_colors, val_colors


def auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")


print(f"{'disease':<48} {'nodes':>6} {'colors':>6} | "
      f"{'node tr/va/te AUC':>22} | {'color tr/va/te AUC':>22}")
for d in DISEASES:
    emb, ci, ny, cy, nsplit, nc = build(d)
    test_colors = np.unique(ci[nsplit == 1])
    train_colors, val_colors = split_colors(nc, test_colors)

    tr_mask = np.isin(ci, train_colors)
    va_mask = np.isin(ci, val_colors)
    te_mask = np.isin(ci, test_colors)

    sc = StandardScaler().fit(emb[tr_mask])
    X = sc.transform(emb)

    # node level - linear
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X[tr_mask], ny[tr_mask])
    pr = clf.predict_proba(X)[:, 1]
    n_tr, n_va, n_te = auc(ny[tr_mask], pr[tr_mask]), auc(ny[va_mask], pr[va_mask]), auc(ny[te_mask], pr[te_mask])

    # node level - nonlinear (gradient boosting) = realistic ceiling on feature signal
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05).fit(X[tr_mask], ny[tr_mask])
    gpr = gb.predict_proba(X)[:, 1]
    g_tr, g_va, g_te = auc(ny[tr_mask], gpr[tr_mask]), auc(ny[va_mask], gpr[va_mask]), auc(ny[te_mask], gpr[te_mask])
    print(f"    [GB node]  train {g_tr:.3f}  val {g_va:.3f}  test {g_te:.3f}")

    # color (patient) level: mean episode embedding per color
    Xc = np.zeros((nc, X.shape[1]))
    for c in range(nc):
        Xc[c] = X[ci == c].mean(0)
    cclf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xc[train_colors], cy[train_colors])
    cp = cclf.predict_proba(Xc)[:, 1]
    c_tr = auc(cy[train_colors], cp[train_colors])
    c_va = auc(cy[val_colors], cp[val_colors])
    c_te = auc(cy[test_colors], cp[test_colors])

    print(f"{d:<48} {len(ci):>6} {nc:>6} | "
          f"{n_tr:>6.3f}/{n_va:.3f}/{n_te:.3f}      | "
          f"{c_tr:>6.3f}/{c_va:.3f}/{c_te:.3f}")
