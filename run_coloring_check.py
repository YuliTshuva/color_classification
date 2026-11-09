"""
Yuli Tshuva.
run_coloring_check.py - does hpo_metab's coloring (I_O: inside / outside, with copies)
work on the other real datasets too? For ppi_bp, hpo_neuro and em_user: the chosen
trial of results/hpo_copies/final.csv with only the coloring changed to I_O, on the
same 10 seeds (0-9). Compared against the coloring the search picked.

Each dataset's seeds are split into chunks; one chunk is one child process on one GPU
(run_hpo_copies.py --run-trial, so the factored copies patches apply). Every chunk
writes its own CSV. Resuming: rerun the same command, only missing seeds are trained.

Usage (server, in cc/ with the venv active):
  python run_coloring_check.py --smoke                 # 1 seed, 15 epochs, results/coloring_check_smoke/
  python run_coloring_check.py --max-parallel 4
  python run_coloring_check.py --collect               # summary.csv only

Results (results/coloring_check/):
  <dataset>/seeds_<a>-<b>.csv   <- one row per seed
  <dataset>/logs/, jobs/
  summary.csv                   <- per dataset: picked coloring vs I_O, val and test mean +- std
"""

# Imports
import run_hpo_copies  # patches run_hpo (factored copies, coloring as a param)
import run_hpo as rh
from train08 import EPOCHS
from os.path import join
import argparse
import json
import os
import pandas as pd
import subprocess
import sys
import time
import torch

SOURCE = join("results", "hpo_copies", "final.csv")
RESULTS = join("results", "coloring_check")
DATASETS = ["hpo_neuro", "em_user", "ppi_bp"]  # longest first
COLORING = "I_O"
SEEDS = list(range(10))
PARAMS = ["aggregation", "pooling", "hidden", "layers", "lr", "dropout", "input_dropout", "batch",
          "weight_decay", "norm", "jk", "head"]
TYPES = {"hidden": int, "layers": int, "head": int, "jk": bool, "lr": float, "dropout": float,
         "input_dropout": float, "batch": float, "weight_decay": float}


def picked_rows():
    final = pd.read_csv(SOURCE, float_precision="round_trip")  # the exact lr / dropout of the trial
    return {r["dataset"]: r for _, r in final.iterrows()}


def params_of(row):
    """The chosen trial's params (plain Python types, for JSON) with the coloring set to I_O."""
    p = {k: TYPES.get(k, str)(row[k]) for k in PARAMS}
    p["coloring"] = COLORING
    return p


def chunks(seeds, n):
    size = -(-len(seeds) // n)
    return [seeds[i:i + size] for i in range(0, len(seeds), size)]


def plan(args, root):
    """[(dataset, seeds, job_path, csv)] for every chunk with missing seeds."""
    rows = picked_rows()
    seeds = SEEDS[:1] if args.smoke else SEEDS
    jobs = []
    for name in DATASETS:
        d = join(root, name)
        os.makedirs(join(d, "jobs"), exist_ok=True)
        os.makedirs(join(d, "logs"), exist_ok=True)
        hps = rh.trial_hps(name, "colors", params_of(rows[name]))
        n = 1 if args.smoke else args.chunks.get(name, 1)
        for part in chunks(seeds, n):
            tag = f"seeds_{part[0]}-{part[-1]}"
            csv = join(d, f"{tag}.csv")
            if os.path.exists(csv):
                df = pd.read_csv(csv, on_bad_lines="skip").dropna(subset=["seed", "val_f1"])
                if set(part) <= set(df["seed"].astype(int)):
                    continue
            job_path = join(d, "jobs", f"{tag}.json")
            with open(job_path, "w") as f:
                json.dump({"dataset": name, "number": tag, "hps": hps, "seeds": part, "csv": csv,
                           "prune_below": None}, f)
            jobs.append((name, part, job_path, csv))
    return jobs


def run_all(args, root, epochs):
    jobs = plan(args, root)
    n_gpus = torch.cuda.device_count()
    slots = [str(g) for g in range(n_gpus)][:args.max_parallel] or [None]
    print(f"{len(jobs)} chunk(s) to run, {len(slots)} at a time", flush=True)
    running, failed = {}, []
    while jobs or running:
        for slot in [i for i in range(len(slots)) if i not in running]:
            if not jobs:
                break
            name, part, job_path, csv = jobs.pop(0)
            env = dict(os.environ)
            if slots[slot] is not None:
                env["CUDA_VISIBLE_DEVICES"] = slots[slot]
            log = open(join(root, name, "logs", f"seeds_{part[0]}-{part[-1]}.log"), "a")
            cmd = [sys.executable, os.path.abspath(run_hpo_copies.__file__), "--run-trial", job_path,
                   "--data-root", args.data_root, "--epochs", str(epochs)]
            running[slot] = (subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log, name, part)
            print(f"  start {name} seeds {part[0]}-{part[-1]} on GPU {slots[slot]}", flush=True)
        time.sleep(5)
        for slot, (proc, log, name, part) in list(running.items()):
            if proc.poll() is None:
                continue
            log.close()
            del running[slot]
            if proc.returncode != 0:
                failed.append(f"{name} seeds {part[0]}-{part[-1]}")
                print(f"  FAILED {name} seeds {part[0]}-{part[-1]} (exit {proc.returncode}) - see its log", flush=True)
            else:
                print(f"  done {name} seeds {part[0]}-{part[-1]}", flush=True)
    if failed:
        print(f"{len(failed)} chunk(s) failed: {failed}; rerun the same command to retry")
    collect(root)


def stats(df):
    return {"n_seeds": len(df),
            "val_mean": round(100 * df["val_f1"].mean(), 2), "val_std": round(100 * df["val_f1"].std(), 2),
            "test_mean": round(100 * df["test_f1"].mean(), 2), "test_std": round(100 * df["test_f1"].std(), 2)}


def collect(root):
    rows = picked_rows()
    out = []
    for name in DATASETS:
        d = join(root, name)
        parts = [pd.read_csv(join(d, f), on_bad_lines="skip") for f in sorted(os.listdir(d))
                 if f.startswith("seeds_") and f.endswith(".csv")] if os.path.isdir(d) else []
        if not parts:
            continue
        df = pd.concat(parts).dropna(subset=["seed", "val_f1", "test_f1"])
        df["seed"] = df["seed"].astype(int)
        df = df.drop_duplicates("seed", keep="last")
        if (df["val_f1"] < 0).any():
            print(f"{name}: OOM")
            continue
        r = rows[name]
        out.append({"dataset": name, "picked_coloring": r["coloring"], "picked_val": r["val_mean"],
                    "picked_test": r["test_mean"], "picked_test_std": r["test_std"], "coloring": COLORING,
                    **stats(df)})
    if out:
        summary = pd.DataFrame(out)
        summary["test_diff"] = (summary["test_mean"] - summary["picked_test"]).round(2)
        summary.to_csv(join(root, "summary.csv"), index=False)
        print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="hpo_metab's coloring (I_O) with each real dataset's chosen hps.")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--data-root", default=join("data", "raw"))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--smoke", action="store_true", help="1 seed, 15 epochs, into results/coloring_check_smoke/")
    parser.add_argument("--collect", action="store_true", help="Only rebuild summary.csv")
    args = parser.parse_args()
    # Seed chunks per dataset (one child each): the slow datasets are split over two GPUs
    args.chunks = {"hpo_neuro": 2, "em_user": 2, "ppi_bp": 1}
    root = RESULTS + ("_smoke" if args.smoke else "")
    if args.collect:
        collect(root)
    else:
        run_all(args, root, 15 if args.smoke else args.epochs)


if __name__ == "__main__":
    main()
