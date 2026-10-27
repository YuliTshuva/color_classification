"""
Statistics for every processed subgraph dataset (data/processed/*), plus the
pregnancy k=20 graph as a reference row.

Writes reports/csv/{summary,thresholds,subgraphs,scaling}.csv and
reports/figures/*.png.

Usage:
  python scripts/compute_stats.py
"""

import glob
import json
import os
import re
import sys
from os.path import join, basename, exists

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ROOT, PROCESSED_DIR, CSV_DIR, FIGURES_DIR

FRACTION_RE = re.compile(r"^(?P<base>.+)_f(?P<pct>\d{3})$")
SYNTHEA_RE = re.compile(r"^synthea_(?P<k>\d+)k$")


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_processed(name):
    d = join(PROCESSED_DIR, name)
    nodes = pd.read_parquet(join(d, "nodes.parquet"), columns=["node_id", "entity_id"])
    ent = pd.read_csv(join(d, "entities.csv"))
    with open(join(d, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    codes = pd.Categorical(nodes["entity_id"], categories=ent["entity_id"])
    return {
        "name": name,
        "entity_of_node": np.asarray(codes.codes, dtype=np.int64),
        "entity_label": ent["label"].to_numpy(),
        "edges": np.load(join(d, "edges.npy")),
        "meta": meta,
    }


def load_pregnancy_reference():
    """data/pregnancy_MiniLM_L6/k_20 read the same way, self-loops dropped."""
    d = join(ROOT, "data", "pregnancy_MiniLM_L6", "k_20")
    if not exists(d):
        return None
    edges = np.loadtxt(join(d, "edges.txt"), dtype=np.int64)
    edges = edges[edges[:, 0] != edges[:, 1]]
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    color = np.loadtxt(join(d, "nodes.txt"), dtype=np.int64)[:, 1]
    labels = np.loadtxt(join(d, "color_labels.txt"), dtype=np.int64)
    labels = labels[np.argsort(labels[:, 0]), 1]
    return {"name": "pregnancy (ours, k=20)", "entity_of_node": color, "entity_label": labels,
            "edges": edges, "meta": {"k": 20}}


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------

def graph_stats(g):
    ent, lab, edges = g["entity_of_node"], g["entity_label"], g["edges"]
    n, n_ent = len(ent), len(lab)
    u, v = edges[:, 0], edges[:, 1]
    e = len(edges)
    deg = np.bincount(np.concatenate([u, v]), minlength=n)

    adj = sp.coo_matrix((np.ones(e, dtype=np.int8), (u, v)), shape=(n, n)).tocsr()
    n_cc, cc = connected_components(adj, directed=False)
    cc_sizes = np.bincount(cc)

    internal = ent[u] == ent[v]
    cross_u, cross_v = ent[u[~internal]], ent[v[~internal]]
    homophily = float((lab[cross_u] == lab[cross_v]).mean()) if len(cross_u) else np.nan
    # Share of cross edges expected to be same-label if entities were paired at
    # random, weighted by entity size (how many node endpoints each label owns)
    p1 = float(lab[ent].mean())
    chance = p1 ** 2 + (1 - p1) ** 2

    sizes = np.bincount(ent, minlength=n_ent)
    int_edges = np.bincount(ent[u[internal]], minlength=n_ent)
    pairs = sizes * (sizes - 1) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        int_density = np.where(pairs > 0, int_edges / pairs, np.nan)

    # Components of each subgraph using its internal edges only
    int_adj = sp.coo_matrix((np.ones(internal.sum(), dtype=np.int8), (u[internal], v[internal])),
                            shape=(n, n)).tocsr()
    _, int_cc = connected_components(int_adj, directed=False)
    comps_per_entity = pd.Series(int_cc).groupby(ent).nunique().reindex(range(n_ent), fill_value=0).to_numpy()
    node_deg_by_entity = np.bincount(ent, weights=deg, minlength=n_ent)

    summary = {
        "dataset": g["name"],
        "nodes": n, "edges": e,
        "avg_degree": 2 * e / n,
        "density": 2 * e / (n * (n - 1)),
        "components": n_cc,
        "largest_component": int(cc_sizes.max()),
        "largest_component_pct": 100 * cc_sizes.max() / n,
        "isolated_nodes": int((deg == 0).sum()),
    }
    has_nodes = sizes > 0
    sub = {
        "dataset": g["name"], "group": "all",
        "entities": int(has_nodes.sum()),
        "positive": int(lab[has_nodes].sum()),
        "positive_pct": 100 * lab[has_nodes].mean(),
        "size_mean": sizes[has_nodes].mean(), "size_std": sizes[has_nodes].std(),
        "size_median": float(np.median(sizes[has_nodes])),
        "size_min": int(sizes[has_nodes].min()), "size_max": int(sizes[has_nodes].max()),
        "internal_density": float(np.nanmean(int_density)),
        "components_per_subgraph": comps_per_entity[has_nodes].mean(),
        "internal_edge_pct": 100 * internal.mean(),
        "cross_label_homophily_pct": 100 * homophily,
        "cross_label_homophily_chance_pct": 100 * chance,
        "mean_node_degree": deg.mean(),
    }
    rows = [sub]
    for label in (1, 0):
        m = has_nodes & (lab == label)
        if not m.any():
            continue
        rows.append({
            "dataset": g["name"], "group": f"label {label}",
            "entities": int(m.sum()),
            "size_mean": sizes[m].mean(), "size_std": sizes[m].std(),
            "size_median": float(np.median(sizes[m])),
            "size_min": int(sizes[m].min()), "size_max": int(sizes[m].max()),
            "internal_density": float(np.nanmean(int_density[m])),
            "components_per_subgraph": comps_per_entity[m].mean(),
            "mean_node_degree": node_deg_by_entity[m].sum() / sizes[m].sum(),
        })
    return summary, rows, sizes[has_nodes]


def overlap_stats(name):
    """
    Statistics for a dataset whose subgraphs overlap (EM-User). Every subgraph
    is measured on its own induced edges. Cross-entity label homophily is left
    out: with shared nodes, an edge can be internal to one subgraph and cross
    to another, so the quantity is not defined the same way.
    """
    d = join(PROCESSED_DIR, name)
    edges = np.load(join(d, "edges.npy"))
    members = pd.read_parquet(join(d, "memberships.parquet"))
    ent = pd.read_csv(join(d, "entities.csv"))
    with open(join(d, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    u, v = edges[:, 0], edges[:, 1]
    n, e = meta["n_nodes"], len(edges)
    deg = np.bincount(np.concatenate([u, v]), minlength=n)
    adj = sp.coo_matrix((np.ones(e, dtype=np.int8), (u, v)), shape=(n, n)).tocsr()
    adj = adj + adj.T
    n_cc, cc = connected_components(adj, directed=False)
    cc_sizes = np.bincount(cc)

    per = []
    internal_keys = []
    for eid, part in members.groupby("entity_id"):
        idx = np.unique(part["node_id"].to_numpy())
        sub = sp.triu(adj[idx][:, idx], k=1).tocoo()
        k = len(idx)
        internal_keys.append(idx[sub.row] * n + idx[sub.col])
        comps = connected_components(adj[idx][:, idx], directed=False)[0]
        per.append({"entity_id": eid, "size": k, "internal_edges": sub.nnz,
                    "density": sub.nnz / (k * (k - 1) / 2) if k > 1 else np.nan,
                    "components": comps, "degree_sum": deg[idx].sum()})
    per = pd.DataFrame(per).merge(ent[["entity_id", "label"]], on="entity_id")
    n_internal = len(np.unique(np.concatenate(internal_keys)))

    summary = {"dataset": name, "nodes": n, "edges": e, "avg_degree": 2 * e / n,
               "density": 2 * e / (n * (n - 1)), "components": n_cc,
               "largest_component": int(cc_sizes.max()), "largest_component_pct": 100 * cc_sizes.max() / n,
               "isolated_nodes": int((deg == 0).sum())}
    rows = [{"dataset": name, "group": "all", "entities": len(per),
             "positive": int(per["label"].sum()), "positive_pct": 100 * per["label"].mean(),
             "size_mean": per["size"].mean(), "size_std": per["size"].std(ddof=0),
             "size_median": per["size"].median(), "size_min": per["size"].min(), "size_max": per["size"].max(),
             "internal_density": per["density"].mean(), "components_per_subgraph": per["components"].mean(),
             "internal_edge_pct": 100 * n_internal / e, "cross_label_homophily_pct": np.nan,
             "cross_label_homophily_chance_pct": np.nan, "mean_node_degree": deg.mean()}]
    for label in (1, 0):
        m = per[per["label"] == label]
        rows.append({"dataset": name, "group": f"label {label}", "entities": len(m),
                     "size_mean": m["size"].mean(), "size_std": m["size"].std(ddof=0),
                     "size_median": m["size"].median(), "size_min": m["size"].min(), "size_max": m["size"].max(),
                     "internal_density": m["density"].mean(), "components_per_subgraph": m["components"].mean(),
                     "mean_node_degree": m["degree_sum"].sum() / m["size"].sum()})
    return summary, rows, per["size"].to_numpy(), meta


def reference_rows():
    """PPI-BP, HPO-Neuro, HPO-Metab as loaded from the SubGNN archive (multi-class)."""
    rows = []
    for name in ["ppi_bp", "hpo_neuro", "hpo_metab", "em_user"]:
        d = join(ROOT, "data", "raw", "subgnn", name)
        if not exists(join(d, "edge_list.txt")):
            continue
        e = pd.read_csv(join(d, "edge_list.txt"), sep=" ", header=None, usecols=[0, 1]).to_numpy(np.int64)
        raw_lines = len(e)
        e = e[e[:, 0] != e[:, 1]]
        e = np.unique(np.sort(e, axis=1), axis=0)
        sub = pd.read_csv(join(d, "subgraphs.pth"), sep="\t", header=None, names=["nodes", "label", "split"])
        sizes = sub["nodes"].str.count("-") + 1
        classes = sorted({c for lab in sub["label"] for c in str(lab).split("-")})
        rows.append({"dataset": name, "nodes": int(e.max()) + 1, "edge_lines": raw_lines,
                     "edges_undirected": len(e), "subgraphs": len(sub), "avg_size": sizes.mean(),
                     "classes": len(classes), "multi_label": bool(sub["label"].str.contains("-").any())})
    return pd.DataFrame(rows)


def size_histogram(name, sizes):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.figure(figsize=(5, 3.2))
    bins = np.unique(np.logspace(0, np.log10(max(sizes.max(), 2)), 40).astype(int))
    plt.hist(sizes, bins=bins, color="#4C72B0", edgecolor="white")
    plt.xscale("log")
    plt.yscale("log")
    plt.gca().xaxis.set_minor_formatter(NullFormatter())
    plt.xlabel("nodes per subgraph")
    plt.ylabel("subgraphs")
    plt.title(name, fontsize=10)
    plt.tight_layout()
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    plt.savefig(join(FIGURES_DIR, f"sizes_{safe}.png"), dpi=150)
    plt.close()


def scaling_rows(names):
    rows = []
    for name in names:
        with open(join(PROCESSED_DIR, name, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        m = FRACTION_RE.match(name)
        base, frac = (m["base"], int(m["pct"]) / 100) if m else (name, 1.0)
        s = SYNTHEA_RE.match(base)
        series = "synthea" if s else base
        if series == "synthea" and frac < 1:
            continue
        rows.append({
            "series": series, "dataset": name, "fraction": frac,
            "entities": meta["n_entities"], "nodes": meta["n_nodes"], "edges": meta["n_edges"],
            "embedding_s": meta["embedding_seconds"], "edges_s": meta["edge_seconds"],
            "build_s": meta["embedding_seconds"] + meta["edge_seconds"],
            "peak_ram_mb": max(meta["embedding_peak_ram_mb"], meta["edge_peak_ram_mb"]),
            "disk_mb": meta["disk_mb"],
        })
    return pd.DataFrame(rows).sort_values(["series", "nodes"])


def scaling_plots(df):
    for col, ylabel, fname in [("edges", "edges (E)", "scaling_edges.png"),
                               ("build_s", "build time (s)", "scaling_time.png")]:
        plt.figure(figsize=(5, 3.5))
        for series, part in df.groupby("series"):
            if len(part) < 2:
                continue
            plt.plot(part["nodes"], part[col], marker="o", label=series)
        plt.xscale("log")
        plt.yscale("log")
        plt.gca().xaxis.set_minor_formatter(NullFormatter())
        plt.xlabel("nodes (N)")
        plt.ylabel(ylabel)
        plt.legend(fontsize=8)
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(join(FIGURES_DIR, fname), dpi=150)
        plt.close()


def main():
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    names = sorted(basename(p) for p in glob.glob(join(PROCESSED_DIR, "*")) if exists(join(p, "meta.json")))
    main_names = [n for n in names if not FRACTION_RE.match(n)]

    ref = load_pregnancy_reference()
    jobs = ([ref["name"]] if ref is not None else []) + main_names

    summaries, subgraph_rows, k_rows = [], [], []
    for name in jobs:
        print(f"stats: {name}")
        if ref is not None and name == ref["name"]:
            summary, rows, sizes = graph_stats(ref)
            meta = ref["meta"]
        elif exists(join(PROCESSED_DIR, name, "memberships.parquet")):
            summary, rows, sizes, meta = overlap_stats(name)
        else:
            g = load_processed(name)
            summary, rows, sizes = graph_stats(g)
            meta = g["meta"]
            del g
        g = {"name": name}
        for key in ["embedding_seconds", "edge_seconds", "embedding_peak_ram_mb", "edge_peak_ram_mb", "disk_mb"]:
            summary[key] = meta.get(key)
        summary["edge_rule"] = meta.get("edge_rule", "kNN k=20")
        summaries.append(summary)
        subgraph_rows.extend(rows)
        size_histogram(g["name"], sizes)
        for k, v in (meta.get("k_sensitivity") or {}).items():
            k_rows.append({"dataset": g["name"], "k": int(k), "edges": v["edges"], "avg_degree": v["avg_degree"]})

    pd.DataFrame(summaries).to_csv(join(CSV_DIR, "summary.csv"), index=False, float_format="%.6g")
    pd.DataFrame(subgraph_rows).to_csv(join(CSV_DIR, "subgraphs.csv"), index=False, float_format="%.6g")
    pd.DataFrame(k_rows).to_csv(join(CSV_DIR, "thresholds.csv"), index=False, float_format="%.6g")
    scaling = scaling_rows([n for n in names if n != "em_user"])
    scaling.to_csv(join(CSV_DIR, "scaling.csv"), index=False, float_format="%.6g")
    scaling_plots(scaling)
    refs = reference_rows()
    refs.to_csv(join(CSV_DIR, "subgnn_reference.csv"), index=False, float_format="%.6g")
    print(refs.to_string(index=False))
    print(pd.DataFrame(summaries).drop(columns=["edge_rule"]).to_string(index=False))
    print(pd.DataFrame(subgraph_rows).to_string(index=False))
    print(scaling.to_string(index=False))


if __name__ == "__main__":
    main()
