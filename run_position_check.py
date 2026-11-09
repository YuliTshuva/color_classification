"""
Yuli Tshuva.
run_position_check.py - two ideas from GPEN on each real dataset's winning config of
results/hpo_copies (copies, free coloring), on the same 10 seeds (0-9):
  pos  : GPEN's global position input. PageRank (damping 0.85, 100 iterations) gives
         each edge the weight R[i] + R[j]; the maximum spanning tree of those weights is
         rooted at the top-PageRank node (each disconnected part at its own top node).
         A node's input gets a fixed one-hot of its depth in that tree, appended to the
         trainable GLASS vectors; a copy gets its node's.
  drop : during training the pooling reads a random 80% of each subgraph's members (at
         least one); validation and test read them all. The dropped members still pass
         messages (rebuilding the copies graph every epoch would be too slow).
  both : pos and drop together.
The winners' own 10 seeds ("base") come from results/hpo_copies, not rerun.

Each (dataset, variant) is split into seed chunks; one chunk is one child process on one
GPU (this script with --run-trial: run_hpo's trial runner with the factored copies
patches and the two above). Each chunk has its own CSV. Resuming: rerun the same
command, only missing seeds are trained.

Usage (server, in cc/ with the venv active):
  python run_position_check.py --smoke                 # 1 seed, 15 epochs, results/position_check_smoke/
  python run_position_check.py --max-parallel 4
  python run_position_check.py --collect               # summary.csv only

Results (results/position_check/):
  positions/<dataset>.pt               <- tree depth per node (shared by smoke and full runs)
  <dataset>/<variant>/seeds_<a>-<b>.csv  <- one row per seed
  <dataset>/logs/, jobs/
  summary.csv                          <- per dataset and variant: val / test / train mean +- std
"""

# Imports
import run_hpo_copies  # noqa: F401 - patches run_hpo (factored copies, coloring as a param)
import run_hpo as rh
import run_gpen_compare as rgc
from train08 import EPOCHS
from os.path import join
import argparse
import copy
import json
import numpy as np
import os
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order, connected_components, minimum_spanning_tree
import subprocess
import sys
import time
import torch
import torch.nn as nn

SOURCE = join("results", "hpo_copies")
RESULTS = join("results", "position_check")
POSITIONS = join(RESULTS, "positions")
DATASETS = ["hpo_neuro", "em_user", "hpo_metab", "ppi_bp"]
SEEDS = list(range(10))
CHUNK = 5  # seeds per child
MEMBER_KEEP = 0.8
VARIANTS = {"pos": {"position": True},
            "drop": {"member_keep": MEMBER_KEEP},
            "both": {"position": True, "member_keep": MEMBER_KEEP}}
PAGERANK_DAMPING, PAGERANK_ITERS = 0.85, 100


# ---------------------------------------------------------------------------
# GPEN's global position (computed once per dataset)
# ---------------------------------------------------------------------------

def tree_depths(edges, n):
    """
    Depth of every node in GPEN's tree: the maximum spanning tree of the edge weights
    R[i] + R[j] (R = PageRank), rooted at the top-PageRank node of each connected part.
    :param edges: [2, E] tensor, each undirected pair once
    :return: (depth int64 array [n], stats dict)
    """
    src, dst = edges[0].numpy(), edges[1].numpy()
    a = sp.coo_matrix((np.ones(2 * len(src)), (np.concatenate([src, dst]), np.concatenate([dst, src]))),
                      shape=(n, n)).tocsr()
    deg = np.asarray(a.sum(1)).ravel()
    # PageRank: follow an edge with probability 0.85, jump anywhere otherwise
    walk = (a @ sp.diags(1.0 / np.maximum(deg, 1))).tocsr()  # column j divided by deg(j)
    r = np.full(n, 1.0 / n)
    for _ in range(PAGERANK_ITERS):
        r = PAGERANK_DAMPING * (walk @ r) + (1 - PAGERANK_DAMPING) / n
    # Maximum spanning tree = minimum spanning tree of (c - w), all positive
    w = r[src] + r[dst]
    c = 2 * w.max()
    tree = minimum_spanning_tree(sp.coo_matrix((c - w, (src, dst)), shape=(n, n)).tocsr())
    tree = (tree + tree.T).tocsr()
    # One extra node linked to the top-PageRank node of every connected part: one BFS gives all depths
    n_parts, part = connected_components(tree, directed=False)
    roots = np.zeros(n_parts, dtype=np.int64)
    best = np.full(n_parts, -1.0)
    for v in range(n):  # n is at most ~57k
        if r[v] > best[part[v]]:
            best[part[v]], roots[part[v]] = r[v], v
    t = tree.tocoo()
    hub = n
    rows = np.concatenate([t.row, np.full(n_parts, hub), roots])
    cols = np.concatenate([t.col, roots, np.full(n_parts, hub)])
    full = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n + 1, n + 1)).tocsr()
    order, pred = breadth_first_order(full, hub, directed=False, return_predecessors=True)
    depth = np.zeros(n + 1, dtype=np.int64)
    for v in order[1:]:
        depth[v] = depth[pred[v]] + 1
    depth = depth[:n] - 1  # the extra node is not a level
    counts = np.bincount(depth)
    stats = {"nodes": n, "parts": int(n_parts), "largest_part": int(np.bincount(part).max()),
             "max_depth": int(depth.max()), "median_depth": float(np.median(depth)),
             "levels_with_1pct": int((counts >= 0.01 * n).sum())}
    return depth, stats


def positions(prep):
    """One-hot tree depth per node, [n_nodes, max_depth + 1], cached in POSITIONS."""
    name = prep["name"]
    path = join(POSITIONS, f"{name}.pt")
    if os.path.exists(path):
        saved = torch.load(path)
    else:
        depth, stats = tree_depths(prep["graph_edges"], prep["n_nodes"])
        saved = {"depth": torch.from_numpy(depth), "stats": stats}
        os.makedirs(POSITIONS, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        torch.save(saved, tmp)
        os.replace(tmp, path)  # children may compute at once; the last write wins, all equal
    depth = saved["depth"]
    assert depth.numel() == prep["n_nodes"], f"{name}: positions for {depth.numel()} nodes, graph has {prep['n_nodes']}"
    return torch.nn.functional.one_hot(depth, int(depth.max()) + 1).float()


# ---------------------------------------------------------------------------
# Patches on run_gpen_compare (active in this process and in its children)
# ---------------------------------------------------------------------------

_build_input, _make_readout = rgc.build_input, rgc.make_gpen_readout


def build_input(prep, hps):
    """run_gpen_compare's input, with the fixed position one-hot in front when hps["position"]."""
    x, module, dim = _build_input(prep, hps)
    if not hps.get("position"):
        return x, module, dim
    pos = positions(prep)
    x = pos if x is None else torch.cat([pos, x], dim=1)  # train() puts x before the module's weight
    return x, module, dim + pos.size(1)


class DropMembers(nn.Module):
    """A readout that, in training, pools a random MEMBER_KEEP of each subgraph's members (at least one)."""

    def __init__(self, inner, keep):
        super().__init__()
        self.inner, self.keep = inner, keep
        self.out_dim = inner.out_dim

    def forward(self, h, color_indices, n_colors, *rest):
        if not self.training or rest:  # attention pooling over member pairs is left as is
            return self.inner(h, color_indices, n_colors, *rest)
        kept = torch.rand(color_indices.numel(), device=h.device) < self.keep
        n_kept = torch.bincount(color_indices[kept], minlength=n_colors)
        kept |= n_kept[color_indices] == 0  # a subgraph that lost every member keeps them all
        return self.inner(h[kept], color_indices[kept], n_colors)


def make_gpen_readout(readout, dim, hps):
    pooling = _make_readout(readout, dim, hps)
    keep = hps.get("member_keep", 1.0)
    if pooling is None or keep >= 1.0:
        return pooling
    return DropMembers(pooling, keep)


rgc.build_input = build_input
rgc.make_gpen_readout = make_gpen_readout


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def winners():
    """Per dataset: (trial, seconds per seed) of the chosen trial in results/hpo_copies/final.csv."""
    final = pd.read_csv(join(SOURCE, "final.csv"))
    return {r["dataset"]: (int(r["trial"]), float(r["seconds_mean"])) for _, r in final.iterrows()}


def base_hps(name, trial):
    with open(join(SOURCE, name, "colors", "jobs", f"{trial}_final.json")) as f:
        return json.load(f)["hps"]


def precompute_positions(data_root):
    """Tree depths of every dataset, before the children start (they only read the cache)."""
    for name in DATASETS:
        path = join(POSITIONS, f"{name}.pt")
        if not os.path.exists(path):
            start = time.time()
            ds = rgc.load_dataset(name, data_root)
            positions({"name": name, "graph_edges": ds["edges"], "n_nodes": ds["n_nodes"]})
            print(f"  positions of {name} in {time.time() - start:.0f}s", flush=True)
        print(f"  {name}: {torch.load(path)['stats']}", flush=True)


def plan(args, root):
    """[(seconds estimate, dataset, variant, seeds, job_path, log_path)] for every chunk with missing seeds."""
    seeds = SEEDS[:1] if args.smoke else SEEDS
    jobs = []
    for name, (trial, secs) in winners().items():
        if name not in DATASETS:
            continue
        d = join(root, name)
        os.makedirs(join(d, "jobs"), exist_ok=True)
        os.makedirs(join(d, "logs"), exist_ok=True)
        for variant, extra in VARIANTS.items():
            hps = {**copy.deepcopy(base_hps(name, trial)), **extra}
            os.makedirs(join(d, variant), exist_ok=True)
            for i in range(0, len(seeds), CHUNK):
                part = seeds[i:i + CHUNK]
                tag = f"seeds_{part[0]}-{part[-1]}"
                csv = join(d, variant, f"{tag}.csv")
                if os.path.exists(csv):
                    df = pd.read_csv(csv, on_bad_lines="skip").dropna(subset=["seed", "val_f1"])
                    if set(part) <= set(df["seed"].astype(int)):
                        continue
                job_path = join(d, "jobs", f"{variant}_{tag}.json")
                with open(job_path, "w") as f:
                    json.dump({"dataset": name, "number": f"{trial}_{variant}", "hps": hps, "seeds": part,
                               "csv": csv, "prune_below": None}, f)
                jobs.append((secs * len(part), name, variant, part, job_path, join(d, "logs", f"{variant}_{tag}.log")))
    return sorted(jobs, key=lambda j: -j[0])  # longest chunks first


def run_all(args, root, epochs):
    precompute_positions(args.data_root)
    jobs = plan(args, root)
    n_gpus = torch.cuda.device_count()
    slots = [str(g) for g in range(n_gpus)][:args.max_parallel] or [None]
    print(f"{len(jobs)} chunk(s) to run, {len(slots)} at a time, ~{sum(j[0] for j in jobs) / 3600:.1f} GPU-hours",
          flush=True)
    running, failed = {}, []
    while jobs or running:
        for slot in [i for i in range(len(slots)) if i not in running]:
            if not jobs:
                break
            _, name, variant, part, job_path, log_path = jobs.pop(0)
            env = dict(os.environ)
            if slots[slot] is not None:
                env["CUDA_VISIBLE_DEVICES"] = slots[slot]
            log = open(log_path, "a")
            cmd = [sys.executable, os.path.abspath(__file__), "--run-trial", job_path,
                   "--data-root", args.data_root, "--epochs", str(epochs)]
            running[slot] = (subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log,
                             f"{name} {variant} seeds {part[0]}-{part[-1]}")
            print(f"  start {running[slot][2]} on GPU {slots[slot]}", flush=True)
        time.sleep(5)
        for slot, (proc, log, what) in list(running.items()):
            if proc.poll() is None:
                continue
            log.close()
            del running[slot]
            if proc.returncode != 0:
                failed.append(what)
                print(f"  FAILED {what} (exit {proc.returncode}) - see its log", flush=True)
            else:
                print(f"  done {what}", flush=True)
    if failed:
        print(f"{len(failed)} chunk(s) failed: {failed}; rerun the same command to retry")
    collect(root)


def read_seeds(paths):
    parts = [pd.read_csv(p, on_bad_lines="skip") for p in paths if os.path.exists(p)]
    if not parts:
        return None
    df = pd.concat(parts).dropna(subset=["seed", "val_f1", "test_f1"])
    df["seed"] = df["seed"].astype(int)
    return df[df["seed"].isin(SEEDS)].drop_duplicates("seed", keep="last")


def collect(root):
    rows = []
    for name, (trial, _) in winners().items():
        if name not in DATASETS:
            continue
        sources = {"base": [join(SOURCE, name, "colors", "trials", f"{trial}.csv")]}
        for variant in VARIANTS:
            d = join(root, name, variant)
            sources[variant] = [join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".csv")] if os.path.isdir(d) else []
        for variant, paths in sources.items():
            df = read_seeds(paths)
            if df is None or df.empty:
                continue
            rows.append({"dataset": name, "trial": trial, "variant": variant, "n_seeds": len(df),
                         "oom": bool((df["val_f1"] < 0).any()),
                         **{f"{k}_{s}": round(100 * getattr(df[f"{k}_f1"], s)(), 2)
                            for k in ("val", "test", "train") for s in ("mean", "std")}})
    if not rows:
        return
    summary = pd.DataFrame(rows)
    base = summary[summary["variant"] == "base"].set_index("dataset")["test_mean"]
    summary["test_vs_base"] = (summary["test_mean"] - summary["dataset"].map(base)).round(2)
    summary.to_csv(join(root, "summary.csv"), index=False)
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="GPEN's position input and member dropping on the real winners.")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--data-root", default=join("data", "raw"))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--smoke", action="store_true", help="1 seed, 15 epochs, into results/position_check_smoke/")
    parser.add_argument("--collect", action="store_true", help="Only rebuild summary.csv")
    parser.add_argument("--run-trial", help=argparse.SUPPRESS)  # internal: one chunk in this process
    args = parser.parse_args()
    if args.run_trial:
        rh.run_trial(args.run_trial, args.data_root, args.epochs)
        return
    root = RESULTS + ("_smoke" if args.smoke else "")
    if args.collect:
        collect(root)
    else:
        run_all(args, root, 15 if args.smoke else args.epochs)


if __name__ == "__main__":
    main()
