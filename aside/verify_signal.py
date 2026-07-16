"""
Independent verification of the 'no generalizable signal' claim.

(1) Edge homophily on the SAVED graph (nodes.txt/edges.txt/color_labels.txt only)
    -> uses exactly what the model trains on; no embeddings, no alignment assumption.
(2) Embedding sanity (NaN/zero/constant) + per-split positive rates.
(3) Permutation control: GB val AUC with shuffled labels vs real labels.
"""
import os
import numpy as np
import pandas as pd
from os.path import join
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

BASE = join(os.path.dirname(__file__), "..", "data", "tamars_datasets")
DISEASES = sorted(os.listdir(BASE))


def saved_graph(disease, k=20):
    d = join(BASE, disease, f"k_{k}")
    nodes = [int(l.split()[1]) for l in open(join(d, "nodes.txt")).read().strip().split("\n")]
    nodes = np.array(nodes)  # color per node
    clab = {}
    for l in open(join(d, "color_labels.txt")).read().strip().split("\n"):
        c, y = l.split()
        clab[int(c)] = int(y)
    node_label = np.array([clab[c] for c in nodes])
    edges = []
    for l in open(join(d, "edges.txt")).read().strip().split("\n"):
        u, v = l.split()
        edges.append((int(u), int(v)))
    edges = np.array(edges)
    return nodes, node_label, edges, clab


print("=== (1) Edge homophily on saved k_20 graph (independent of embeddings) ===")
print(f"{'disease':<48} {'pos_rate':>8} {'homophily':>10} {'expected':>9} {'lift':>6}")
for d in DISEASES:
    nodes, ny, edges, clab = saved_graph(d)
    p = ny.mean()
    same = (ny[edges[:, 0]] == ny[edges[:, 1]]).mean()
    # expected same-label rate under random pairing: p^2 + (1-p)^2
    exp = p ** 2 + (1 - p) ** 2
    print(f"{d:<48} {p:>8.3f} {same:>10.3f} {exp:>9.3f} {same-exp:>+6.3f}")

print("\n=== (2)+(3) Embedding sanity + permutation control (k=20 split logic) ===")


def build(disease):
    folder = join(BASE, disease)
    train_emb = np.load(join(folder, "train_embeddings.npy"))
    test_emb = np.load(join(folder, "test_embeddings.npy"))
    train_df = pd.read_csv(join(folder, "data", "train", "listfile.csv"))
    test_df = pd.read_csv(join(folder, "data", "test", "listfile.csv"))
    emb = np.vstack([train_emb, test_emb])
    df = pd.concat([train_df, test_df], ignore_index=True)
    node_split = np.array([0] * len(train_emb) + [1] * len(test_emb))
    valid = [el.split("_")[0] for el in df["stay"] if "episode2" in el]
    valid = [s for s in valid if f"{s}_episode1_timeseries.csv" in df["stay"].values]
    df["stay_id"] = df["stay"].apply(lambda x: x.split("_")[0])
    keep = df.index[df["stay_id"].isin(valid)]
    emb = emb[keep]; node_split = node_split[keep]
    df = df[df["stay_id"].isin(valid)].reset_index(drop=True)
    stay_labels = df.groupby("stay_id")["y_true"].max().to_dict()
    df["label"] = df["stay_id"].apply(lambda x: stay_labels[x])
    ci = pd.factorize(df["stay_id"])[0]
    return emb, ci, df["label"].values.astype(int), node_split


for d in DISEASES[:3]:
    emb, ci, ny, nsplit = build(d)
    test_colors = np.unique(ci[nsplit == 1])
    tv = np.array([c for c in np.unique(ci) if c not in set(test_colors.tolist())])
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(tv)); nval = max(1, int(len(tv) * 0.2))
    val_c, tr_c = tv[perm[:nval]], tv[perm[nval:]]
    trm, vam = np.isin(ci, tr_c), np.isin(ci, val_c)

    nan = np.isnan(emb).any(); zero = (np.abs(emb).sum(1) == 0).sum(); const = (emb.std(0) == 0).sum()
    X = StandardScaler().fit(emb[trm]).transform(emb)

    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05).fit(X[trm], ny[trm])
    real_val = roc_auc_score(ny[vam], gb.predict_proba(X[vam])[:, 1])
    # permutation control: shuffle TRAIN labels, refit, eval on real val labels
    yp = ny[trm].copy(); rng.shuffle(yp)
    gbp = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05).fit(X[trm], yp)
    perm_val = roc_auc_score(ny[vam], gbp.predict_proba(X[vam])[:, 1])

    print(f"{d:<48} pos tr/va={ny[trm].mean():.3f}/{ny[vam].mean():.3f} "
          f"| nan={nan} zerorows={zero} constdims={const} "
          f"| GB val real={real_val:.3f} shuffled={perm_val:.3f}")
