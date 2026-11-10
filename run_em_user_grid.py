"""
Yuli Tshuva.
run_em_user_grid.py - em_user with the one coloring (I_O: inside / outside, with copies),
on the part of the space its search never tried. Report 11's 69.0 is I_O with the
hyperparameters tuned for the clique coloring (sum aggregation, gated attention pooling,
3 layers of 32); the search itself tried I_O in 1 trial of 160. In run_edge_colors_copies
that combination is the weak one without a clique (57-68), and gcn / mean aggregation
with mean / size pooling gives 83-90.

The grid (coloring fixed to I_O, 1 layer, 10 seeds each, 16 configs):
  aggregation : gcn, mean
  pooling     : mean, size
  hidden      : 64, 128
  training    : glass - the one-coloring config's training, untouched (report 11:
                        mini-batches of 12, GLASS's stopping, layer norm, its lr / dropouts)
                full  - run_edge_colors_copies' 86.9 model's training (full batch,
                        no norm, lr 1e-3, dropout 0.5, node loss on, stop on validation)
Everything else of each training's source config is kept as it ran.

One config is one child process on one GPU (run_hpo_copies.py --run-trial, so the
factored copies patches apply), with its own CSV. Resuming: rerun the same command,
only missing seeds are trained. The test set is never used to pick anything.

Usage (server, in cc/ with the venv active):
  python run_em_user_grid.py --smoke                 # 1 seed, 15 epochs, 2 configs, results/em_user_grid_smoke/
  python run_em_user_grid.py --max-parallel 4
  python run_em_user_grid.py --collect               # summary.csv only

Results (results/em_user_grid/):
  runs/<config>.csv   <- one row per seed
  logs/, jobs/
  summary.csv         <- per config: val / test / train mean +- std, best validation first
                         ("base" = the one-coloring config of report 11, when its runs are there)
"""

# Imports
from os.path import join
import argparse
import copy
import json
import os
import pandas as pd
import subprocess
import sys
import time
import torch

DATASET = "em_user"
RESULTS = join("results", "em_user_grid")
BASE = join("results", "coloring_check", DATASET)  # the one-coloring config's 10 seeds
CHILD = join(os.path.dirname(os.path.abspath(__file__)), "run_hpo_copies.py")
SEEDS = list(range(10))
COLORING = [["I"], ["O"]]
AGGREGATIONS = ["gcn", "mean"]
POOLINGS = ["mean", "size"]
HIDDEN = [64, 128]
SECONDS_PER_SEED = {"glass": 150, "full": 45}  # rough, for the ordering and the estimate only

# The one-coloring config, exactly as run_coloring_check ran it on em_user
GLASS_BASE = {"coloring": COLORING, "copies": True, "copy_clique": False, "backbone": "rgcn", "num_heads": 4,
              "graph_agg": "sum", "readout": "gated_attn", "gnn_embedding_dim": 32, "gnn_hidden_dim": 32,
              "k_gnn_layers": 3, "jk": True, "gnn_mlp_hidden_dims": 64, "node_input": "glass_ft",
              "feature_norm": "none", "attn_pool_hidden_dim": 32, "attn_pool_dropout": 0.0, "pma_heads": 4,
              "pma_sabs": 1, "batch_size": 12, "lr": 0.0019240274976922488, "lr_on": "train", "lr_factor": 0.7,
              "glass_stop": True, "norm": "layer", "input_norm": "graph", "input_dropout": 0.5,
              "msg_dropout": 0.30000000000000004, "color_mlp_hidden_dims": 64,
              "gnn_dropout_rate": 0.30000000000000004, "mlp_dropout_rate": 0.0, "weight_decay": 1e-05,
              "node_loss_weight": 0.0}
# run_edge_colors_copies' best em_user config (Icopy_Iplain_O__ours__gcn__mean__h128_lr1_d0.5_c0, 86.9),
# with the coloring set to I_O
FULL_BASE = {"coloring": COLORING, "copies": True, "copy_clique": False, "backbone": "rgcn", "num_heads": 4,
             "graph_agg": "gcn", "readout": "mean", "gnn_embedding_dim": 128, "gnn_hidden_dim": 128,
             "k_gnn_layers": 1, "jk": True, "gnn_mlp_hidden_dims": 64, "node_input": "glass_ft",
             "feature_norm": "none", "attn_pool_hidden_dim": 32, "attn_pool_dropout": 0.0, "pma_heads": 4,
             "pma_sabs": 1, "lr": 0.001, "n_batches": 1, "norm": None, "gnn_dropout_rate": 0.5,
             "mlp_dropout_rate": 0.0, "weight_decay": 0.0, "color_mlp_hidden_dims": 64, "node_loss_weight": 1.0,
             "batch_size": None, "lr_on": "val", "glass_stop": False, "input_norm": None, "input_dropout": 0.0,
             "msg_dropout": 0.0}
TRAININGS = {"glass": GLASS_BASE, "full": FULL_BASE}  # slowest first


def configs():
    """{label: hps} of the 16 grid points."""
    out = {}
    for training, base in TRAININGS.items():
        for agg in AGGREGATIONS:
            for pool in POOLINGS:
                for hidden in HIDDEN:
                    out[f"{training}__{agg}__{pool}__h{hidden}"] = {
                        **copy.deepcopy(base), "graph_agg": agg, "readout": pool, "gnn_embedding_dim": hidden,
                        "gnn_hidden_dim": hidden, "k_gnn_layers": 1}
    return out


def read_seeds(paths):
    parts = [pd.read_csv(p, on_bad_lines="skip") for p in paths if os.path.exists(p)]
    if not parts:
        return None
    df = pd.concat(parts).dropna(subset=["seed", "val_f1", "test_f1"])
    df["seed"] = df["seed"].astype(int)
    return df[df["seed"].isin(SEEDS)].drop_duplicates("seed", keep="last")


def plan(args, root):
    """[(label, seeds, job_path, log_path)] for every config with missing seeds."""
    seeds = SEEDS[:1] if args.smoke else SEEDS
    for sub in ("runs", "jobs", "logs"):
        os.makedirs(join(root, sub), exist_ok=True)
    grid = configs()
    if args.smoke:  # one config per training
        grid = {k: v for k, v in grid.items() if k.endswith("__gcn__mean__h64")}
    jobs = []
    for label, hps in grid.items():
        csv = join(root, "runs", f"{label}.csv")
        df = read_seeds([csv])
        if df is not None and set(seeds) <= set(df["seed"]):
            continue
        job_path = join(root, "jobs", f"{label}.json")
        with open(job_path, "w") as f:
            json.dump({"dataset": DATASET, "number": label, "hps": hps, "seeds": seeds, "csv": csv,
                       "prune_below": None}, f)
        jobs.append((label, seeds, job_path, join(root, "logs", f"{label}.log")))
    return jobs


def run_all(args, root):
    if not os.path.exists(CHILD):
        sys.exit(f"{CHILD} is missing - the children run it")
    jobs = plan(args, root)
    n_gpus = torch.cuda.device_count()
    slots = [str(g) for g in range(n_gpus)][:args.max_parallel] or [None]
    gpu_hours = sum(len(seeds) * SECONDS_PER_SEED[label.split("__")[0]] for label, seeds, _, _ in jobs) / 3600
    print(f"{len(jobs)} config(s) to run, {len(slots)} at a time, ~{gpu_hours:.1f} GPU-hours", flush=True)
    running, failed = {}, []
    while jobs or running:
        for slot in [i for i in range(len(slots)) if i not in running]:
            if not jobs:
                break
            label, seeds, job_path, log_path = jobs.pop(0)
            env = dict(os.environ)
            if slots[slot] is not None:
                env["CUDA_VISIBLE_DEVICES"] = slots[slot]
            log = open(log_path, "a")
            cmd = [sys.executable, CHILD, "--run-trial", job_path, "--data-root", args.data_root]
            if args.smoke or args.epochs is not None:
                cmd += ["--epochs", str(15 if args.smoke else args.epochs)]
            running[slot] = (subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log, label)
            print(f"  start {label} on GPU {slots[slot]}", flush=True)
        time.sleep(5)
        for slot, (proc, log, label) in list(running.items()):
            if proc.poll() is None:
                continue
            log.close()
            del running[slot]
            if proc.returncode != 0:
                failed.append(label)
                print(f"  FAILED {label} (exit {proc.returncode}) - see its log", flush=True)
            else:
                print(f"  done {label}", flush=True)
    if failed:
        print(f"{len(failed)} config(s) failed: {failed}; rerun the same command to retry")
    collect(root)


def collect(root):
    sources = {label: [join(root, "runs", f"{label}.csv")] for label in configs()}
    if os.path.isdir(BASE):
        sources["base"] = [join(BASE, f) for f in sorted(os.listdir(BASE))
                           if f.startswith("seeds_") and f.endswith(".csv")]
    rows = []
    for label, paths in sources.items():
        df = read_seeds(paths)
        if df is None or df.empty:
            continue
        training, agg, pool, hidden = label.split("__") if label != "base" else ("glass", "sum", "gated_attn", "h32")
        rows.append({"config": label, "training": training, "aggregation": agg, "pooling": pool,
                     "hidden": int(hidden[1:]), "n_seeds": len(df), "oom": bool((df["val_f1"] < 0).any()),
                     **{f"{k}_{s}": round(100 * getattr(df[f"{k}_f1"], s)(), 2)
                        for k in ("val", "test", "train") for s in ("mean", "std")},
                     "epochs_mean": round(df["epochs"].mean(), 1), "seconds_mean": round(df["seconds"].mean(), 1)})
    if not rows:
        print("no finished seeds yet")
        return
    # Best validation first; the test column is only read, never used to order
    summary = pd.DataFrame(rows).sort_values(["val_mean", "config"], ascending=[False, True])
    summary.to_csv(join(root, "summary.csv"), index=False)
    print(summary.drop(columns=["training", "aggregation", "pooling", "hidden"]).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="em_user, coloring I_O: aggregation x pooling x hidden x training.")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--data-root", default=join("data", "raw"))
    parser.add_argument("--epochs", type=int, default=None, help="Epoch cap (default: the children's own)")
    parser.add_argument("--smoke", action="store_true",
                        help="1 seed, 15 epochs, one config per training, into results/em_user_grid_smoke/")
    parser.add_argument("--collect", action="store_true", help="Only rebuild summary.csv")
    args = parser.parse_args()
    root = RESULTS + ("_smoke" if args.smoke else "")
    if args.collect:
        collect(root)
    else:
        run_all(args, root)


if __name__ == "__main__":
    main()
