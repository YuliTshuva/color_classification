"""
Test error against subgraph size, one line per case (report 08, idea 4).

Reads results/<case dir>/<dataset>/test_predictions.csv, written by
run_gpen_compare.py's run_seeds. The test subgraphs of each dataset are split
into size bins with about the same number of subgraphs each (quartiles, fewer
when sizes repeat). The error of a bin is the share of its test subgraphs
labeled wrongly, over all seeds; for multi-label hpo_neuro "wrong" means the
predicted label set is not exactly the true one. The bars are the std of that
share across seeds.

Output:
  results/error_vs_size/<dataset>.png   one plot per dataset
  results/error_vs_size/all.png         every dataset, one panel each
  results/error_vs_size/error_vs_size.csv   the plotted numbers

Usage (from the project root):
  python scripts/plot_error_vs_size.py
  python scripts/plot_error_vs_size.py --bins 5
"""

import argparse
import os
from os.path import join

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASETS = ["ppi_bp", "hpo_metab", "hpo_neuro", "em_user", "density", "cut_ratio", "coreness", "component"]
# case -> (results folder, legend name). Colors are the categorical slots in fixed order.
CASES = {"pool": ("gpen_compare_v2", "plain clique (mean)"),
         "no_clique": ("gpen_compare_no_clique", "no clique"),
         "clique_sym": ("gpen_compare_clique_sym", "clique, symmetric"),
         "clique_sym_w": ("gpen_compare_clique_sym_w", "clique, symmetric + 1/N_C"),
         "hub_graph": ("gpen_compare_hub_graph", "hub graph")}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]  # identity is never color alone
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def load(dataset):
    """{case: predictions DataFrame} for the cases that have predictions for this dataset."""
    out = {}
    for case, (folder, _) in CASES.items():
        path = join("results", folder, dataset, "test_predictions.csv")
        if os.path.exists(path):
            out[case] = pd.read_csv(path, keep_default_na=False)
    return out


def size_bins(sizes, n_bins):
    """Bin edges with about the same number of test subgraphs per bin."""
    edges = np.unique(np.quantile(sizes, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        edges = np.array([sizes.min(), sizes.max() + 1])
    return edges


def bin_of(sizes, edges):
    """Bin i holds sizes in [edges[i], edges[i+1]); the last bin also holds the largest size."""
    return np.clip(np.searchsorted(edges, sizes, side="right") - 1, 0, len(edges) - 2)


def bin_errors(df, edges):
    """Per bin: mean error over seeds, its std across seeds, and the number of test subgraphs."""
    df = df.assign(bin=bin_of(df["size"].to_numpy(), edges))
    per_seed = 1 - df.groupby(["bin", "seed"])["correct"].mean()
    stats = per_seed.groupby("bin").agg(["mean", "std"]).fillna(0)
    n = df[df["seed"] == df["seed"].min()].groupby("bin").size()
    return stats.join(n.rename("n_subgraphs"))


def draw(ax, dataset, preds, n_bins):
    """One dataset's lines on ax; returns the plotted rows."""
    # The bins come from the test subgraphs themselves, the same for every case
    sizes = next(iter(preds.values())).drop_duplicates("subgraph")["size"].to_numpy()
    edges = size_bins(sizes, n_bins)
    # Labels show the smallest and largest test subgraph actually in each bin
    bins = bin_of(sizes, edges)
    labels = [f"{sizes[bins == b].min()}-{sizes[bins == b].max()}" if (bins == b).any() else "-"
              for b in range(len(edges) - 1)]
    x = np.arange(len(labels))
    rows = []
    for k, case in enumerate(CASES):
        if case not in preds:
            continue
        st = bin_errors(preds[case], edges).reindex(range(len(labels)))
        ax.errorbar(x, st["mean"], yerr=st["std"], color=COLORS[k], marker=MARKERS[k], markersize=6,
                    linewidth=2, capsize=3, label=CASES[case][1])
        for b, r in st.iterrows():
            rows.append({"dataset": dataset, "case": case, "size_bin": labels[b], "error": round(r["mean"], 4),
                         "error_std": round(r["std"], 4), "n_test_subgraphs": int(r["n_subgraphs"]) if r["n_subgraphs"] == r["n_subgraphs"] else 0,
                         "n_seeds": preds[case]["seed"].nunique()})
    ax.set_xticks(x, labels)
    ax.set_title(dataset, color=INK, fontsize=11)
    ax.set_xlabel("subgraph size (nodes)", color=MUTED)
    ax.set_ylabel("test error", color=MUTED)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Test error against subgraph size, per case.")
    parser.add_argument("--bins", type=int, default=4, help="Size bins per dataset")
    args = parser.parse_args()
    out_dir = join("results", "error_vs_size")
    os.makedirs(out_dir, exist_ok=True)

    found = {d: p for d in DATASETS if (p := load(d))}
    if not found:
        print("No test_predictions.csv under results/ - run run_gpen_compare.py first")
        return

    rows = []
    for dataset, preds in found.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        rows += draw(ax, dataset, preds, args.bins)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(join(out_dir, f"{dataset}.png"), dpi=150)
        plt.close(fig)
        print(f"{dataset}: {', '.join(preds)}")

    n = len(found)
    cols = min(n, 4)
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(4.2 * cols, 3.4 * np.ceil(n / cols)),
                             squeeze=False)
    for ax, (dataset, preds) in zip(axes.flat, found.items()):
        draw(ax, dataset, preds, args.bins)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    # One legend for all panels, from every panel in case some lack a case
    legend = {}
    for ax in axes.flat[:n]:
        for h, name in zip(*ax.get_legend_handles_labels()):
            legend.setdefault(name, h)
    names = [CASES[c][1] for c in CASES if CASES[c][1] in legend]
    handles = [legend[name] for name in names]
    fig.legend(handles, names, loc="lower center", ncol=len(names), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(join(out_dir, "all.png"), dpi=150)
    plt.close(fig)

    pd.DataFrame(rows).to_csv(join(out_dir, "error_vs_size.csv"), index=False)
    print(f"Saved under {out_dir}/")


if __name__ == "__main__":
    main()
