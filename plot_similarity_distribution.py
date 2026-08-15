"""
Yuli Tshuva.
plot_similarity_distribution.py - the all-pairs cosine similarity distribution.

Report 03 states that the pregnancy MiniLM embeddings are severely anisotropic:
two *random* nodes sit at cosine +0.979, so the k=20 neighbourhood of exp08 is
decided in the fourth decimal place. That claim was made from a sample; this
script measures it exactly.

Every one of the N(N-1)/2 pairs is scored, in row blocks, and accumulated into a
fixed histogram over [-1, 1] - the N x N matrix is never materialised. The same
is done in the three similarity spaces of graph_construction.project, so the
"centering removes the common component" claim is visible rather than asserted.

Overlaid on each distribution is the similarity of the *edges the graph actually
keeps* (the k=20 nearest neighbours of exp08), which is what makes the point:
the kept edges are indistinguishable from random pairs in the raw space.

Usage:
  python plot_similarity_distribution.py
  python plot_similarity_distribution.py --spaces raw --bins 20000
"""

import argparse
from os.path import join

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from graph_construction import SIM_SPACES, project, load_candidates

# dataviz reference palette, light mode. Slots 1-3 validate all-pairs.
COLORS = {"raw": "#2a78d6", "centered": "#eb6834", "whitened": "#1baf7a"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"

OUT_DIR = "plots"
LO, HI = -1.0, 1.0


def all_pair_histogram(emb, bins, chunk=2048, verbose=True):
    """
    Exact histogram of the cosine similarity of all N(N-1)/2 unordered pairs.

    Row blocks are multiplied against the full matrix and only the strict upper
    triangle of each block is counted, so every pair is seen once and no node is
    compared to itself.

    :param emb: [N, d] float tensor.
    :param bins: number of equal-width bins spanning [-1, 1].
    :return: (counts [bins] int64, edges [bins + 1] float64, mean float)
    """
    x = torch.nn.functional.normalize(emb.to(torch.float32), p=2, dim=1)
    n = x.size(0)
    scale = bins / (HI - LO)

    counts = np.zeros(bins, dtype=np.int64)
    total, running_sum = 0, 0.0

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = (x[start:stop] @ x.T).numpy()  # [c, N]

        # Strict upper triangle: keep column j only where j > i
        cols = np.arange(n)[None, :]
        rows = np.arange(start, stop)[:, None]
        vals = block[cols > rows]

        idx = np.clip(((vals - LO) * scale).astype(np.int64), 0, bins - 1)
        counts += np.bincount(idx, minlength=bins)
        total += vals.size
        running_sum += float(vals.astype(np.float64).sum())

        if verbose and (start // chunk) % 5 == 0:
            print(f"  {stop}/{n} rows", flush=True)

    return counts, np.linspace(LO, HI, bins + 1), running_sum / total


def quantiles_from_hist(counts, edges, qs):
    """Quantiles read off the histogram; accurate to one bin width."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    cum = np.cumsum(counts) / counts.sum()
    return [float(centers[np.searchsorted(cum, q)]) for q in qs]


def main():
    parser = argparse.ArgumentParser(description="All-pairs cosine similarity distribution.")
    parser.add_argument("--dataset", type=str, default="pregnancy")
    parser.add_argument("--name", type=str, default="pregnancy")
    parser.add_argument("--spaces", type=str, nargs="+", default=SIM_SPACES, choices=SIM_SPACES)
    parser.add_argument("--bins", type=int, default=20000, help="Bin width 2/bins over [-1, 1]")
    parser.add_argument("--k", type=int, default=20, help="Neighbourhood size to overlay")
    parser.add_argument("--k-max", type=int, default=50, help="Candidate cache to read")
    args = parser.parse_args()

    from train08 import load_pregnancy_node_features
    from utils import load_pregnancy_data

    features = load_pregnancy_node_features()
    *_, split = load_pregnancy_data(k=20)
    fit_mask = split == 0
    n = features.size(0)
    print(f"{n} nodes, {features.size(1)} dims, {n * (n - 1) // 2:,} pairs")

    stats = {}
    for space in args.spaces:
        print(f"\n{space}:")
        counts, edges, mean = all_pair_histogram(project(features, space, fit_mask), args.bins)
        p1, p50, p99 = quantiles_from_hist(counts, edges, [0.01, 0.5, 0.99])

        # What the k=20 graph of exp08 actually kept, for the same nodes
        _, nbr_sim = load_candidates(args.dataset, args.name, args.k_max, space=space)
        kept = nbr_sim[:, :args.k].numpy().ravel()

        stats[space] = dict(counts=counts, edges=edges, mean=mean, kept=kept,
                            p1=p1, p50=p50, p99=p99)
        print(f"  random pair: mean {mean:.4f}, p1 {p1:.4f}, median {p50:.4f}, p99 {p99:.4f}")
        print(f"  k={args.k} edges: mean {kept.mean():.4f}, min {kept.min():.4f}")

    # ----------------------------------------------------------------- plot
    # A single panel still has to be wide enough for the title and legend above it
    width = max(8.0, 4.6 * len(args.spaces))
    fig, axes = plt.subplots(1, len(args.spaces), figsize=(width, 4.0), facecolor=SURFACE)
    axes = np.atleast_1d(axes)

    for ax, space in zip(axes, args.spaces):
        s = stats[space]
        centers = 0.5 * (s["edges"][:-1] + s["edges"][1:])
        density = s["counts"] / (s["counts"].sum() * (s["edges"][1] - s["edges"][0]))
        color = COLORS[space]

        # Window the x-axis on where the mass actually is
        lo, hi = s["p1"], max(s["p99"], s["kept"].max())
        pad = 0.08 * (hi - lo)
        lo, hi = lo - pad, min(hi + pad, 1.0)
        m = (centers >= lo) & (centers <= hi)

        ax.fill_between(centers[m], density[m], color=color, alpha=0.16, linewidth=0)
        ax.plot(centers[m], density[m], color=color, linewidth=2,
                label="all pairs (random)")

        kd, ke = np.histogram(s["kept"], bins=200, range=(lo, hi), density=True)
        kc = 0.5 * (ke[:-1] + ke[1:])
        ax.plot(kc, kd, color=INK, linewidth=1.6, linestyle=(0, (4, 2)),
                label=f"edges kept by k={args.k}")

        # Log density: in the raw space almost all the mass is in a spike a few
        # thousandths wide, and a linear axis shows nothing but that spike.
        ax.set_yscale("log")
        ax.set_ylim(bottom=max(1e-4, density[density > 0].min()))
        ax.set_facecolor(SURFACE)
        # ax.set_title(f"{space}    mean {s['mean']:+.4f}", color=INK, fontsize=15,
        #              loc="center", pad=10)
        ax.set_xlabel("cosine similarity", color=MUTED, fontsize=13)
        ax.set_xlim(lo, hi)
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=8))
        ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterMathtext(base=10))
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.grid(axis="both", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("#c3c2b7")
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.tick_params(colors=MUTED, labelsize=8)

    axes[0].set_ylabel("share of pairs (log)", color=MUTED, fontsize=13)

    # One legend for the whole figure: the two marks mean the same in every panel,
    # and a per-panel box would sit on top of the curve it explains.
    handles = [plt.Line2D([], [], color=MUTED, linewidth=2),
               plt.Line2D([], [], color=INK, linewidth=1.6, linestyle=(0, (4, 2)))]
    fig.legend(handles, ["all pairs (a random pair of visits)",
                         f"only the pairs the k={args.k} graph connects"],
               frameon=False, fontsize=15, labelcolor=MUTED, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, 0.835))

    fig.suptitle(f"Cosine similarity of all {n * (n - 1) // 2 / 1e9:.2f} billion node pairs\n"
                 f"({args.name}, {features.size(1)}-dim MiniLM, mean {s['mean']:+.4f})",
                 color=INK, fontsize=20, ha="center", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.89))

    out = join(OUT_DIR, "similarity_distribution.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    fig.savefig(out.replace(".png", ".pdf"), facecolor=SURFACE)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
