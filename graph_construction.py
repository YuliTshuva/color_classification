"""
Yuli Tshuva.
graph_construction.py - Channel 2 of the post-exp08 proposals: the graph.

The exp08 pipeline feeds the relational GNN a kNN graph at a fixed k=20, built
once by data/pregnancy_data_to_graph.py before training ever sees a label. Every
node therefore gets exactly 20 neighbours whether or not it has 20 genuinely
similar ones: nodes in tight clusters lose real neighbours, nodes in sparse
regions gain 20 arbitrary ones.

This module builds the *hybrid* alternative - a small kNN core plus a similarity
threshold:

    core  : the k_core nearest neighbours of every node, always kept. Guarantees
            connectivity and a minimum degree, so no node is ever isolated.
    extra : every further candidate whose cosine similarity exceeds a threshold
            tau. Degree now adapts to local density.

tau is parameterised as a *quantile* q of the eligible candidate similarities
rather than as a raw cosine value, which makes the sweep dataset-independent and
gives the knob clean endpoints:

    q = 0  ->  keep every candidate            (= plain kNN at k_max)
    q = 1  ->  keep nothing beyond the core    (= plain kNN at k_core)

so the baseline density (mean degree 20) sits at roughly q = 1 - 20/k_max.

Candidates
----------
A pure threshold graph would need all N^2 pairs (3.7e9 for pregnancy). Instead we
precompute the k_max nearest neighbours of every node once, cache them, and apply
the threshold inside that candidate set. The only edges this can miss are those
of a node that has more than k_max neighbours above tau; `build_hybrid_edges`
reports how often that cap binds (`saturated_frac`), so the approximation is
measured rather than assumed.

Similarity space
----------------
Cosine over the *raw* embeddings is what data/pregnancy_data_to_graph.py used, so
the raw candidate cache reproduces the exp08 graph exactly (`--verify` checks
this). But those embeddings are severely anisotropic - on pregnancy the mean
cosine between two *random* nodes is +0.979, and the entire k=20 neighbourhood is
decided by differences in the fourth decimal place:

    space      random-pair cosine    kNN sim spread (p1..p99)
    raw              +0.9791                 0.0086
    centered         +0.0065                 0.3894
    whitened         +0.0006                 0.3212

Removing the common component changes the neighbourhoods substantially (top-20
overlap with raw: 27% centered, 12% whitened), so the space is swept as its own
axis rather than fixed. The statistics are fitted on the training split only, the
same convention train08.normalize_features uses.

This is also why tau is a quantile: a raw-cosine threshold would have to be tuned
inside [0.991, 0.9998], whereas the quantile is rank-based and behaves identically
in every space.

Usage (once per dataset, before launching any studies):
  python graph_construction.py --dataset pregnancy --k-max 50
  python graph_construction.py --dataset pregnancy --k-max 50 --verify --describe
  python graph_construction.py --dataset diseases --k-max 50
"""

import argparse
import os
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F

import constants

SIM_SPACES = ["raw", "centered", "whitened"]


def dataset_dir(dataset, name="pregnancy"):
    """Folder holding a dataset's embeddings and its k_* graph folders."""
    if dataset == "diseases":
        return join("data", "tamars_datasets", name)
    return constants.PREGNANCY_DIR


def candidate_path(dataset, name, k_max, space):
    return join(dataset_dir(dataset, name), f"candidates_k{k_max}_cosine_{space}.npz")


# --------------------------------------------------------------------------- #
# Similarity space
# --------------------------------------------------------------------------- #

def project(features, space, fit_mask=None):
    """
    Map the embeddings into the space the similarity is measured in.

    raw       : untouched - reproduces the exp08 graph.
    centered  : subtract the mean embedding. Removes the common component that
                puts every pair at cosine ~0.98.
    whitened  : ZCA - center, then decorrelate and equalise the variance of every
                direction, so no single dominant direction drives the similarity.

    Statistics are fitted on `fit_mask` (the training split) only, matching
    train08.normalize_features, so no test information enters the graph.
    """
    if space == "raw":
        return features
    if fit_mask is None:
        fit_mask = torch.ones(features.size(0), dtype=torch.bool)

    x = features.to(torch.float64)
    mean = x[fit_mask].mean(dim=0, keepdim=True)
    x = x - mean
    if space == "centered":
        return x.to(torch.float32)
    if space == "whitened":
        fit = x[fit_mask]
        cov = (fit.T @ fit) / max(1, fit.size(0) - 1)
        evals, evecs = torch.linalg.eigh(cov)
        w = evecs @ torch.diag(evals.clamp(min=1e-8) ** -0.5) @ evecs.T
        return (x @ w).to(torch.float32)
    raise ValueError(f"Unknown similarity space: {space}")


# --------------------------------------------------------------------------- #
# Candidate neighbours
# --------------------------------------------------------------------------- #

def cosine_candidates(emb, k_max, device=None, chunk=1024, verbose=True):
    """
    Exact top-k_max cosine neighbours of every node, computed in row blocks so the
    N x N similarity matrix is never materialised.

    Self is masked explicitly rather than assumed to be the first hit, which
    matters when embeddings contain exact duplicates.

    :param emb: [N, d] float tensor of raw embeddings.
    :return: (idx [N, k_max] int64, sim [N, k_max] float32), sorted by decreasing
             similarity, both on the CPU.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n = emb.size(0)
    k_max = min(k_max, n - 1)

    embn = F.normalize(emb.to(torch.float32), p=2, dim=1).to(device)

    idx_out = torch.empty(n, k_max, dtype=torch.long)
    sim_out = torch.empty(n, k_max, dtype=torch.float32)

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = embn[start:stop] @ embn.T  # [c, N]

        # Mask self-similarity by position, not by rank
        rows = torch.arange(stop - start, device=device)
        block[rows, torch.arange(start, stop, device=device)] = -float("inf")

        sim, idx = block.topk(k_max, dim=1)
        idx_out[start:stop] = idx.cpu()
        sim_out[start:stop] = sim.cpu()

        if verbose and (start // chunk) % 20 == 0:
            print(f"  candidates: {stop}/{n} nodes", flush=True)

    return idx_out, sim_out


def load_candidates(dataset, name, k_max, space="raw", features=None, fit_mask=None,
                    device=None, rebuild=False):
    """
    Load the cached candidate neighbours, computing and caching them on first use.

    :param features: [N, d] raw embeddings; only needed when the cache is cold.
    :param fit_mask: training-split mask used to fit the space's statistics.
    :return: (idx [N, k_max] int64, sim [N, k_max] float32)
    """
    path = candidate_path(dataset, name, k_max, space)

    if os.path.exists(path) and not rebuild:
        cached = np.load(path)
        return torch.from_numpy(cached["idx"]).long(), torch.from_numpy(cached["sim"]).float()

    if features is None:
        raise FileNotFoundError(
            f"No candidate cache at {path} and no features given. Build it first:\n"
            f"  python graph_construction.py --dataset {dataset} --k-max {k_max}")

    print(f"Building candidate neighbours for {name} (k_max={k_max}, space={space})...", flush=True)
    idx, sim = cosine_candidates(project(features, space, fit_mask), k_max, device=device)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, idx=idx.numpy().astype(np.int32), sim=sim.numpy())
    print(f"Cached to {path}", flush=True)
    return idx, sim


# --------------------------------------------------------------------------- #
# Hybrid graph
# --------------------------------------------------------------------------- #

def _canonical_unique(src, dst, n_nodes):
    """
    Deduplicate undirected pairs. edges.txt stores each pair once and the
    RGCN/RGAT layers symmetrise internally, so we keep the same convention.

    :return: (edge_index [2, E], keys [E] int64) with keys = min * n + max.
    """
    if src.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    lo = torch.minimum(src, dst).to(torch.int64)
    hi = torch.maximum(src, dst).to(torch.int64)
    keys = torch.unique(lo * n_nodes + hi)
    return torch.stack([keys // n_nodes, keys % n_nodes]), keys


def build_hybrid_edges(nbr_idx, nbr_sim, k_core, quantile):
    """
    The k_core-core + similarity-threshold graph.

    tau is the `quantile` of the *eligible* candidate similarities (the ones past
    the core), so quantile=0 keeps the full k_max graph and quantile=1 collapses
    to the pure k_core graph.

    :param nbr_idx: [N, k_max] candidate neighbours, decreasing similarity.
    :param nbr_sim: [N, k_max] their cosine similarities.
    :return: (core_edges [2, E_core], extra_edges [2, E_extra], stats dict).
             The two edge sets are disjoint: a pair that is core for either
             endpoint counts as core.
    """
    n_nodes, k_max = nbr_idx.shape
    if not 1 <= k_core <= k_max:
        raise ValueError(f"k_core must be in [1, {k_max}], got {k_core}")

    src = torch.arange(n_nodes).unsqueeze(1).expand(n_nodes, k_max)

    core_edges, core_keys = _canonical_unique(
        src[:, :k_core].reshape(-1), nbr_idx[:, :k_core].reshape(-1), n_nodes)

    eligible_sim = nbr_sim[:, k_core:]
    if eligible_sim.numel() == 0:
        tau = float("inf")
        extra_mask = torch.zeros_like(eligible_sim, dtype=torch.bool)
    else:
        # numpy: torch.quantile caps input at 2**24 elements
        tau = float(np.quantile(eligible_sim.numpy(), quantile))
        extra_mask = eligible_sim > tau

    extra_edges, extra_keys = _canonical_unique(
        src[:, k_core:][extra_mask], nbr_idx[:, k_core:][extra_mask], n_nodes)

    # A pair kept by the threshold may already be core for the other endpoint
    if extra_keys.numel() and core_keys.numel():
        keep = ~torch.isin(extra_keys, core_keys)
        extra_edges = extra_edges[:, keep]

    n_core, n_extra = core_edges.size(1), extra_edges.size(1)
    degrees = torch.bincount(torch.cat([core_edges, extra_edges], dim=1).reshape(-1),
                             minlength=n_nodes)
    stats = {
        "tau": round(tau, 4),
        "n_core_edges": n_core,
        "n_extra_edges": n_extra,
        "n_edges": n_core + n_extra,
        "mean_degree": round(float(degrees.float().mean()), 2),
        "max_degree": int(degrees.max()),
        "isolated_nodes": int((degrees == 0).sum()),
        # How often the k_max cap binds: nodes whose every candidate survived the
        # threshold would have had more neighbours in a true epsilon-graph.
        "saturated_frac": round(float(extra_mask.all(dim=1).float().mean()), 4),
    }
    return core_edges, extra_edges, stats


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #

def _read_edge_file(path):
    keys = set()
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            u, v = map(int, line.split())
            keys.add((min(u, v), max(u, v)))
    return keys


def verify(dataset, name, k_max, k_check=20):
    """
    Rebuild the k_check kNN graph from the candidate cache and compare it to the
    edges.txt written by data/pregnancy_data_to_graph.py. A near-perfect overlap
    says the cache uses the same metric, the same node ordering and the same
    self-exclusion as the graph exp08 was trained on.

    Exact equality is not expected: the original builder assumes self is the
    nearest hit, which is false for exactly-duplicated embeddings. On pregnancy
    this costs ~0.2% of the edges (jaccard 0.998).

    Only meaningful for space="raw" - the other spaces are supposed to differ.
    """
    idx, sim = load_candidates(dataset, name, k_max, space="raw")
    n_nodes = idx.size(0)

    src = torch.arange(n_nodes).unsqueeze(1).expand(n_nodes, k_check)
    rebuilt, _ = _canonical_unique(src.reshape(-1), idx[:, :k_check].reshape(-1), n_nodes)
    rebuilt = {(int(u), int(v)) for u, v in rebuilt.T.tolist()}

    path = join(dataset_dir(dataset, name), f"k_{k_check}", "edges.txt")
    if not os.path.exists(path):
        print(f"  no {path} to compare against - skipping")
        return
    original = _read_edge_file(path)

    inter = len(rebuilt & original)
    union = len(rebuilt | original)
    print(f"  rebuilt k={k_check}: {len(rebuilt)} edges, original: {len(original)} edges")
    print(f"  overlap {inter} | jaccard {inter / union:.6f} | "
          f"only-rebuilt {len(rebuilt - original)} | only-original {len(original - rebuilt)}")
    if inter / union < 0.99:
        print("  WARNING: low overlap - the candidate cache does not match the exp08 graph.")


def describe(dataset, name, k_max, spaces=SIM_SPACES, k_cores=(2,),
             quantiles=(0.2, 0.4, 0.5, 0.6, 0.8, 0.9)):
    """Print the density the sweep will actually explore, for sanity."""
    for space in spaces:
        idx, sim = load_candidates(dataset, name, k_max, space=space)
        pcts = np.percentile(sim.numpy(), [1, 50, 99])
        print(f"\n{name} / {space}: {idx.size(0)} nodes, candidate similarity "
              f"p1 {pcts[0]:.4f} / median {pcts[1]:.4f} / p99 {pcts[2]:.4f} "
              f"(spread {pcts[2] - pcts[0]:.4f})")
        print(f"{'k_core':>7} {'q':>5} {'tau':>8} {'edges':>10} {'mean deg':>9} "
              f"{'max deg':>8} {'isolated':>9} {'saturated':>10}")
        for k_core in k_cores:
            for q in quantiles:
                _, _, s = build_hybrid_edges(idx, sim, k_core, q)
                print(f"{k_core:>7} {q:>5.2f} {s['tau']:>8.4f} {s['n_edges']:>10} "
                      f"{s['mean_degree']:>9.2f} {s['max_degree']:>8} {s['isolated_nodes']:>9} "
                      f"{s['saturated_frac']:>10.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute and inspect the candidate neighbours for the exp10 hybrid graph.")
    parser.add_argument("--dataset", type=str, default="pregnancy",
                        help="diseases, pregnancy, or another data/ folder with the pregnancy "
                             "layout (e.g. pregnancy_C for the anti-collapse embeddings)")
    parser.add_argument("--names", type=str, nargs="+", default=None,
                        help="Disease names; defaults to all of them for --dataset diseases.")
    parser.add_argument("--k-max", type=int, default=50)
    parser.add_argument("--spaces", type=str, nargs="+", default=SIM_SPACES, choices=SIM_SPACES)
    parser.add_argument("--rebuild", action="store_true", help="Recompute even if cached")
    parser.add_argument("--verify", action="store_true", help="Compare the raw top-20 against edges.txt")
    parser.add_argument("--describe", action="store_true", help="Print the density of the sweep grid")
    args = parser.parse_args()

    # Imported here so the module stays importable without the training stack
    from train08 import (DISEASES, load_disease_node_features, load_pregnancy_node_features,
                         resolve_dataset)
    from utils import load_disease_data, load_pregnancy_data

    args.dataset = resolve_dataset(args.dataset)
    names = args.names or (DISEASES if args.dataset == "diseases" else ["pregnancy"])
    for name in names:
        print(f"\n=== {name} ===")
        features = (load_disease_node_features(name) if args.dataset == "diseases"
                    else load_pregnancy_node_features())
        # The centered/whitened spaces fit their statistics on the training split only
        if args.dataset == "diseases":
            *_, split = load_disease_data(disease=name, k=20)
        else:
            *_, split = load_pregnancy_data(k=20)
        fit_mask = split == 0

        for space in args.spaces:
            load_candidates(args.dataset, name, args.k_max, space=space, features=features,
                            fit_mask=fit_mask, rebuild=args.rebuild)
        if args.verify and "raw" in args.spaces:
            verify(args.dataset, name, args.k_max)
        if args.describe:
            describe(args.dataset, name, args.k_max, spaces=args.spaces)


if __name__ == "__main__":
    main()
