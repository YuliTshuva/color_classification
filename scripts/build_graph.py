"""
Convert a raw dataset into the common subgraph format under
data/processed/<dataset>[_f<pct>]/:

  nodes.parquet     node_id, entity_id, timestamp, text (when there is text)
  embeddings.npy    float32 [N, d], L2-normalized
  edges.npy         int64 [E, 2], u < v, undirected, deduplicated, no self-loops
  edge_weights.npy  float32 cosine similarity per edge
  entities.csv      entity_id, label, split, n_nodes
  meta.json         source, decisions, k, model, build time, peak RAM

Edges: every node links to its K most similar other nodes by cosine (FAISS
exact inner-product search on normalized vectors, in chunks, so nothing N x N
is ever held), then the list is made undirected and deduplicated. This is the
rule used for data/pregnancy_MiniLM_L6/k_20, see config.py.

Usage:
  python scripts/build_graph.py --dataset jodie_wikipedia
  python scripts/build_graph.py --dataset jodie_reddit --fraction 0.25
  python scripts/build_graph.py --dataset synthea_1k
  python scripts/build_graph.py --dataset em_user
"""

import argparse
import glob
import json
import os
import re
import sys
import threading
import time
from os.path import join, exists, getsize

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (RAW_DIR, PROCESSED_DIR, EMBEDDING_MODEL, EMBED_BATCH_SIZE, K, EXTRA_KS,
                    SEED, SPLIT_FRACTIONS, EXACT_MAX_NODES, IVF_NPROBE)


# ----------------------------------------------------------------------------
# Measuring
# ----------------------------------------------------------------------------

class PeakRAM:
    """Samples this process's resident memory in a thread; .peak is in bytes."""

    def __init__(self, interval=0.05):
        self.interval = interval
        self.proc = psutil.Process()

    def __enter__(self):
        self.peak = self.proc.memory_info().rss
        self.start = time.perf_counter()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(self.interval)

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)
        self.seconds = time.perf_counter() - self.start


# ----------------------------------------------------------------------------
# Shared pieces
# ----------------------------------------------------------------------------

def l2_normalize(x):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def embed_texts(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    emb = model.encode(list(texts), batch_size=EMBED_BATCH_SIZE, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=True)
    return emb.astype(np.float32)


def make_index(emb):
    """
    Exact inner-product index up to EXACT_MAX_NODES, IVF above it (exact search
    is quadratic: ~12 h for 1.3M x 384 on this machine).
    :return: (index, description)
    """
    import faiss
    n, d = emb.shape
    if n <= EXACT_MAX_NODES:
        index = faiss.IndexFlatIP(d)
        index.add(emb)
        return index, "exact (IndexFlatIP)"
    nlist = int(4 * np.sqrt(n))
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    rng = np.random.default_rng(SEED)
    index.train(emb[rng.choice(n, size=min(n, 100 * nlist), replace=False)])
    index.add(emb)
    index.nprobe = IVF_NPROBE
    return index, f"approximate (IndexIVFFlat, nlist={nlist}, nprobe={IVF_NPROBE})"


def knn_recall(index, emb, k, n_queries=2000):
    """Share of the true top-k (by similarity) that the index returns, on a random sample of nodes."""
    import faiss
    rng = np.random.default_rng(SEED)
    q = emb[rng.choice(len(emb), size=min(n_queries, len(emb)), replace=False)]
    exact = faiss.IndexFlatIP(emb.shape[1])
    exact.add(emb)
    D_true, _ = exact.search(q, k + 1)
    D_got, I_got = index.search(q, k + 1)
    # Compare by similarity, not by id, so ties between duplicate vectors do
    # not count as misses: a hit is a returned neighbour at least as similar
    # as the true (k+1)-th one (k neighbours plus the node itself)
    hits = (D_got >= D_true[:, -1:] - 1e-6) & (I_got >= 0)
    return float(hits.sum(axis=1).clip(max=k + 1).mean() / (k + 1))


def knn_search(emb, k, chunk=20000):
    """
    Top-k neighbours (self excluded) for every row of an L2-normalized matrix,
    queried in chunks.
    :return: (neighbour ids [N, k], similarities [N, k], search description)
    """
    n, d = emb.shape
    k = min(k, n - 1)
    index, how = make_index(emb)
    ids = np.empty((n, k), dtype=np.int64)
    sims = np.empty((n, k), dtype=np.float32)
    for s in range(0, n, chunk):
        q = emb[s:s + chunk]
        D, I = index.search(q, k + 1)
        rows = np.arange(s, s + len(q))
        # Drop the node itself; exact duplicates may put another node at rank
        # 0, so remove self by id, not by position.
        is_self = I == rows[:, None]
        keep = ~is_self
        # Rows where self was not returned keep their first k hits
        no_self = ~is_self.any(axis=1)
        keep[no_self, k] = False
        ids[s:s + len(q)] = I[keep].reshape(len(q), k)
        sims[s:s + len(q)] = D[keep].reshape(len(q), k)
    return ids, sims, how, index


def knn_edges(ids, sims, k):
    """Undirected, deduplicated, self-loop-free edges from the first k neighbours."""
    n = ids.shape[0]
    src = np.repeat(np.arange(n, dtype=np.int64), k)
    dst = ids[:, :k].reshape(-1)
    w = sims[:, :k].reshape(-1)
    ok = (dst >= 0) & (dst != src)
    src, dst, w = src[ok], dst[ok], w[ok]
    u, v = np.minimum(src, dst), np.maximum(src, dst)
    key = u * n + v
    key, first = np.unique(key, return_index=True)
    return np.stack([key // n, key % n], axis=1), w[first].astype(np.float32)


def split_entities(labels, seed=SEED):
    """70/15/15 by entity, stratified by label."""
    rng = np.random.default_rng(seed)
    split = np.empty(len(labels), dtype=object)
    labels = np.asarray(labels)
    for lab in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == lab))
        n_train = int(round(SPLIT_FRACTIONS[0] * len(idx)))
        n_val = int(round(SPLIT_FRACTIONS[1] * len(idx)))
        split[idx[:n_train]] = "train"
        split[idx[n_train:n_train + n_val]] = "val"
        split[idx[n_train + n_val:]] = "test"
    return split


def subsample_entities(entity_ids, labels, fraction, seed=SEED):
    """Keep a label-stratified random fraction of entities."""
    if fraction >= 1:
        return np.asarray(entity_ids)
    rng = np.random.default_rng(seed)
    entity_ids, labels = np.asarray(entity_ids), np.asarray(labels)
    keep = []
    for lab in np.unique(labels):
        ids = entity_ids[labels == lab]
        keep.append(rng.choice(ids, size=max(1, int(round(fraction * len(ids)))), replace=False))
    return np.concatenate(keep)


def out_name(dataset, fraction):
    return dataset if fraction >= 1 else f"{dataset}_f{int(round(fraction * 100)):03d}"


def dir_size(path):
    return sum(getsize(f) for f in glob.glob(join(path, "*")) if os.path.isfile(f))


def write_dataset(name, nodes, emb, entity_labels, meta, embed_cost, extra_edges=None):
    """
    Build the kNN edges (unless extra_edges is given) and write every output file.
    :param nodes: DataFrame with entity_id and optionally timestamp / text, one row per node
    :param emb: L2-normalized float32 [N, d], or None for datasets with given edges
    :param entity_labels: Series entity_id -> 0/1
    :param embed_cost: (seconds, peak bytes) of producing emb
    :param extra_edges: (edges, weights) for datasets whose edges are not rebuilt
    """
    out = join(PROCESSED_DIR, name)
    os.makedirs(out, exist_ok=True)
    nodes = nodes.reset_index(drop=True)
    nodes.insert(0, "node_id", np.arange(len(nodes), dtype=np.int64))

    if extra_edges is None:
        with PeakRAM() as edge_cost:
            ids, sims, search, index = knn_search(emb, max([K] + EXTRA_KS))
            edges, weights = knn_edges(ids, sims, K)
        meta["knn_search"] = search
        meta["knn_recall_at_k"] = 1.0 if search.startswith("exact") else round(knn_recall(index, emb, K), 4)
        del index
        # Edge counts at the other k values, for the sensitivity table
        k_table = {}
        for k in sorted(set([K] + EXTRA_KS)):
            e = edges if k == K else knn_edges(ids, sims, k)[0]
            k_table[str(k)] = {"edges": int(len(e)), "avg_degree": 2 * len(e) / len(nodes)}
        del ids, sims
        np.save(join(out, "embeddings.npy"), emb)
        edge_seconds, edge_peak = edge_cost.seconds, edge_cost.peak
    else:
        edges, weights = extra_edges
        k_table, edge_seconds, edge_peak = None, 0.0, 0

    np.save(join(out, "edges.npy"), edges.astype(np.int64))
    np.save(join(out, "edge_weights.npy"), weights.astype(np.float32))
    nodes.to_parquet(join(out, "nodes.parquet"), index=False)

    counts = nodes["entity_id"].value_counts()
    ent = pd.DataFrame({"entity_id": entity_labels.index, "label": entity_labels.values.astype(int)})
    ent["split"] = split_entities(ent["label"].values)
    ent["n_nodes"] = ent["entity_id"].map(counts).fillna(0).astype(int)
    ent.to_csv(join(out, "entities.csv"), index=False)

    meta.update({
        "name": name,
        "n_nodes": int(len(nodes)),
        "n_edges": int(len(edges)),
        "n_entities": int(len(ent)),
        "edge_rule": meta.get("edge_rule", f"kNN k={K} on cosine, undirected, deduplicated"),
        "k": None if extra_edges is not None else K,
        "k_sensitivity": k_table,
        "embedding_model": meta.get("embedding_model", EMBEDDING_MODEL),
        "embedding_seconds": round(embed_cost[0], 2),
        "embedding_peak_ram_mb": round(embed_cost[1] / 2 ** 20, 1),
        "edge_seconds": round(edge_seconds, 2),
        "edge_peak_ram_mb": round(edge_peak / 2 ** 20, 1),
        "seed": SEED,
    })
    meta["disk_mb"] = round(dir_size(out) / 2 ** 20, 1)
    with open(join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    meta["disk_mb"] = round(dir_size(out) / 2 ** 20, 1)
    with open(join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {out}: N={len(nodes):,} E={len(edges):,} entities={len(ent):,} "
          f"embed {embed_cost[0]:.1f}s, edges {edge_seconds:.1f}s, disk {meta['disk_mb']} MB")


# ----------------------------------------------------------------------------
# JODIE (Reddit, Wikipedia)
# ----------------------------------------------------------------------------

def build_jodie(which, fraction):
    path = join(RAW_DIR, f"jodie_{which}", f"{which}.csv")
    with PeakRAM() as cost:
        # Header is "user_id,item_id,timestamp,state_label,comma_separated_list_of_features",
        # so the rows have more fields than the header: read without it.
        df = pd.read_csv(path, header=None, skiprows=1, engine="pyarrow")
        df.columns = ["user", "item", "timestamp", "state_label"] + \
                     [f"f{i}" for i in range(df.shape[1] - 4)]
        user_label = df.groupby("user")["state_label"].max().astype(int)
        keep = subsample_entities(user_label.index.values, user_label.values, fraction)
        df = df[df["user"].isin(keep)].sort_values(["user", "timestamp"], kind="stable")
        feats = df.filter(regex=r"^f\d+$").to_numpy(dtype=np.float32)
        mu, sd = feats.mean(0), feats.std(0)
        constant_dims = int((sd == 0).sum())
        feats = (feats - mu) / np.where(sd == 0, 1.0, sd)
        emb = l2_normalize(feats)
        del feats
    nodes = pd.DataFrame({"entity_id": df["user"].values, "timestamp": df["timestamp"].values,
                          "item_id": df["item"].values})
    labels = user_label.loc[np.sort(keep)]
    meta = {
        "source": "https://github.com/claws-lab/jodie (http://snap.stanford.edu/jodie/"
                  f"{which}.csv)",
        "node": "one interaction (Reddit post / Wikipedia edit)",
        "entity": "user",
        "label": "1 if any interaction of the user has state_label == 1 (banned)",
        "text": "none; node vectors are the 172-dim LIWC features, z-scored per "
                "dimension then L2-normalized",
        "constant_feature_dims": constant_dims,
        "embedding_model": "LIWC features (given)",
        "fraction": fraction,
    }
    write_dataset(out_name(f"jodie_{which}", fraction), nodes, emb, labels, meta,
                  (cost.seconds, cost.peak))


# ----------------------------------------------------------------------------
# Synthea
# ----------------------------------------------------------------------------

# Target condition: which patients are label 1, and what is scrubbed from the
# visit text so the label cannot be read off it. A text item (encounter reason,
# observation, procedure, medication, immunization, diagnosis) is dropped
# whole when its description matches `text_pattern`.
SYNTHEA_TARGETS = {
    "diabetes": {
        # Any condition naming diabetes, prediabetes excluded. Matching on the
        # name and not only on 44054006 (Diabetes mellitus type 2) matters:
        # Synthea exports 10 years of history, so a patient diagnosed earlier
        # keeps only complications such as "Disorder of kidney due to diabetes
        # mellitus" (9.2% of patients vs 7.2% with the 44054006 row, p1000).
        "label_pattern": re.compile(r"(?<!pre)diabet", re.IGNORECASE),
        "text_pattern": re.compile(
            r"diabet|prediabet|hyperglyc|insulin|hemoglobin a1c|a1c|metformin|glipizide|glyburide|"
            r"glimepiride|liraglutide|semaglutide|dulaglutide|exenatide|canagliflozin|"
            r"empagliflozin|dapagliflozin|sitagliptin|saxagliptin|linagliptin|alogliptin|"
            r"pioglitazone|repaglinide|nateglinide|acarbose",
            re.IGNORECASE),
    },
}


def _items_by_encounter(df, desc_col="DESCRIPTION"):
    """encounter id -> list of unique descriptions, in first-seen order."""
    df = df[["ENCOUNTER", desc_col]].dropna().drop_duplicates()
    return df.groupby("ENCOUNTER", sort=False)[desc_col].agg(list)


def build_synthea(size, fraction, target="diabetes"):
    tag = f"{size // 1000}k"
    csv = join(RAW_DIR, "synthea", f"p{size}", "csv")
    spec = SYNTHEA_TARGETS[target]
    pattern = spec["text_pattern"]
    with PeakRAM() as cost:
        patients = pd.read_csv(join(csv, "patients.csv"), usecols=["Id"])
        cond = pd.read_csv(join(csv, "conditions.csv"), usecols=["PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION"])
        label_rows = cond["DESCRIPTION"].astype(str).str.contains(spec["label_pattern"])
        positive = set(cond.loc[label_rows, "PATIENT"])
        label_conditions = sorted(cond.loc[label_rows, "DESCRIPTION"].unique())
        labels = pd.Series(patients["Id"].isin(positive).astype(int).values, index=patients["Id"].values)
        keep = subsample_entities(labels.index.values, labels.values, fraction)
        labels = labels.loc[np.sort(keep)]
        keep = set(keep)

        enc = pd.read_csv(join(csv, "encounters.csv"),
                          usecols=["Id", "START", "PATIENT", "ENCOUNTERCLASS", "DESCRIPTION", "REASONDESCRIPTION"])
        enc = enc[enc["PATIENT"].isin(keep)].sort_values(["PATIENT", "START"], kind="stable").reset_index(drop=True)

        # Observations go last: they are long lists of lab names and MiniLM
        # reads only the first 256 tokens
        sections = [("Diagnoses", cond)]
        for name, file in [("Procedures", "procedures.csv"), ("Medications", "medications.csv"),
                           ("Immunizations", "immunizations.csv"), ("Observations", "observations.csv")]:
            sections.append((name, pd.read_csv(join(csv, file), usecols=["PATIENT", "ENCOUNTER", "DESCRIPTION"])))

        removed = {"Encounter reason": 0}
        grouped = []
        for name, df in sections:
            df = df[df["PATIENT"].isin(keep)]
            hit = df["DESCRIPTION"].astype(str).str.contains(pattern)
            removed[name] = int(hit.sum())
            grouped.append((name, _items_by_encounter(df[~hit])))

        def reason(r):
            if isinstance(r, str) and not pattern.search(r):
                return r
            return None

        reasons = enc["REASONDESCRIPTION"].map(reason)
        removed["Encounter reason"] = int((enc["REASONDESCRIPTION"].notna() & reasons.isna()).sum())
        texts = []
        for i, row in enumerate(enc.itertuples(index=False)):
            parts = [f"{row.ENCOUNTERCLASS} encounter: {row.DESCRIPTION}."]
            if isinstance(reasons.iat[i], str):
                parts.append(f"Reason: {reasons.iat[i]}.")
            for name, items in grouped:
                got = items.get(row.Id)
                if got:
                    parts.append(f"{name}: {'; '.join(got)}.")
            texts.append(" ".join(parts))
        texts = pd.Series(texts)
        # Many visits share the exact same text (e.g. routine check-ups), so
        # embed each distinct text once
        uniq, inverse = np.unique(texts.values, return_inverse=True)
        emb = embed_texts(uniq)[inverse]

    nodes = pd.DataFrame({"entity_id": enc["PATIENT"].values, "timestamp": enc["START"].values,
                          "encounter_id": enc["Id"].values, "text": texts.values})
    meta = {
        "source": f"Synthea (github.com/synthetichealth/synthea, master-branch-latest jar), "
                  f"-p {size} -s 42, Massachusetts, default 10 years of history",
        "node": "one encounter",
        "entity": "patient",
        "label": f"1 if any exported condition of the patient matches /{spec['label_pattern'].pattern}/ "
                 f"({target}, prediabetes excluded)",
        "label_conditions": label_conditions,
        "text": "built per encounter: class + description, reason, then the descriptions of the "
                "diagnoses, procedures, medications, immunizations and observations recorded in "
                "that encounter (no values); MiniLM truncates at 256 tokens. Synthea's clinical notes were not used: they repeat "
                "the patient's full history and name in every visit and link to visits only by date.",
        "leakage_pattern": pattern.pattern,
        "leakage_items_removed": removed,
        "unique_texts": int(len(uniq)),
        "fraction": fraction,
    }
    write_dataset(out_name(f"synthea_{tag}", fraction), nodes, emb, labels, meta, (cost.seconds, cost.peak))


def synthea_conditions(size, top=25):
    """Print the most common conditions (by number of patients) in a population."""
    csv = join(RAW_DIR, "synthea", f"p{size}", "csv")
    n = len(pd.read_csv(join(csv, "patients.csv"), usecols=["Id"]))
    cond = pd.read_csv(join(csv, "conditions.csv"), usecols=["PATIENT", "CODE", "DESCRIPTION"])
    counts = cond.drop_duplicates(["PATIENT", "CODE"]).groupby(["CODE", "DESCRIPTION"]).size()
    counts = counts.sort_values(ascending=False).head(top).reset_index(name="patients")
    counts["pct"] = (100 * counts["patients"] / n).round(1)
    print(f"{n} patients\n{counts.to_string(index=False)}")


# ----------------------------------------------------------------------------
# SubGNN EM-User (structural reference, loaded as-is)
# ----------------------------------------------------------------------------

def read_subgnn(name):
    """Global edges (undirected, deduplicated, no self-loops) and the subgraph table."""
    d = join(RAW_DIR, "subgnn", name)
    e = pd.read_csv(join(d, "edge_list.txt"), sep=" ", header=None, usecols=[0, 1]).to_numpy(np.int64)
    e = e[e[:, 0] != e[:, 1]]
    edges = np.unique(np.sort(e, axis=1), axis=0)
    sub = pd.read_csv(join(d, "subgraphs.pth"), sep="\t", header=None, names=["nodes", "label", "split"])
    sub["nodes"] = [np.array(s.split("-"), dtype=np.int64) for s in sub["nodes"]]
    return edges, sub, int(edges.max()) + 1


def build_em_user(fraction):
    if fraction < 1:
        raise SystemExit("em_user is a fixed reference graph; --fraction is not supported")
    out = join(PROCESSED_DIR, "em_user")
    os.makedirs(out, exist_ok=True)
    with PeakRAM() as cost:
        edges, sub, n = read_subgnn("em_user")
    # Subgraphs overlap (a node can belong to several users), so membership is
    # a separate table instead of an entity_id column on the nodes
    sub["entity_id"] = np.arange(len(sub))
    members = sub[["entity_id", "nodes"]].explode("nodes").rename(columns={"nodes": "node_id"})
    members["node_id"] = members["node_id"].astype(np.int64)
    members.to_parquet(join(out, "memberships.parquet"), index=False)
    pd.DataFrame({"node_id": np.arange(n, dtype=np.int64)}).to_parquet(join(out, "nodes.parquet"), index=False)
    np.save(join(out, "edges.npy"), edges)
    ent = pd.DataFrame({"entity_id": sub["entity_id"], "label": (sub["label"] == "female").astype(int),
                        "split": sub["split"], "n_nodes": sub["nodes"].map(len)})
    ent.to_csv(join(out, "entities.csv"), index=False)
    meta = {
        "source": "SubGNN real-world datasets (github.com/mims-harvard/SubGNN, Dropbox SubGNN_data.zip), em_user/",
        "node": "workout", "entity": "user", "label": "gender: female = 1, male = 0",
        "edge_rule": "given by SubGNN, not rebuilt (no text)",
        "splits": "SubGNN's own train/val/test split, kept as-is",
        "overlapping_subgraphs": True,
        "nodes_in_any_subgraph": int(members["node_id"].nunique()),
        "name": "em_user", "n_nodes": n, "n_edges": int(len(edges)), "n_entities": int(len(ent)),
        "embedding_model": None, "k": None, "k_sensitivity": None,
        "embedding_seconds": 0.0, "embedding_peak_ram_mb": 0.0,
        "edge_seconds": round(cost.seconds, 2), "edge_peak_ram_mb": round(cost.peak / 2 ** 20, 1),
    }
    meta["disk_mb"] = round(dir_size(out) / 2 ** 20, 1)
    with open(join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {out}: N={n:,} E={len(edges):,} subgraphs={len(ent)}")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

BUILDERS = {
    "jodie_wikipedia": lambda f: build_jodie("wikipedia", f),
    "jodie_reddit": lambda f: build_jodie("reddit", f),
    "synthea_1k": lambda f: build_synthea(1000, f),
    "synthea_5k": lambda f: build_synthea(5000, f),
    "synthea_20k": lambda f: build_synthea(20000, f),
    "em_user": build_em_user,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Fraction of entities to keep (label-stratified)")
    args = parser.parse_args()
    if args.dataset not in BUILDERS:
        raise SystemExit(f"Unknown dataset {args.dataset}; choose from {sorted(BUILDERS)}")
    BUILDERS[args.dataset](args.fraction)


if __name__ == "__main__":
    main()
