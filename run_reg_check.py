"""
Yuli Tshuva.
run_reg_check.py - are em_user and ppi_bp below GLASS / GPEN because they overfit? (train
F1 ~99.8 on both). The 3 finalists of results/hpo_copies on each, with stronger
regularization, on the same 10 seeds (0-9):
  drop : message dropout and input dropout +0.2 (at most 0.8)
  wd   : weight decay x10, at least 1e-3
  both : drop and wd together
The finalists' own 10 seeds ("base") come from results/hpo_copies, not rerun.
Questions: does test rise toward GLASS, and does validation predict test better?

Each (dataset, finalist, variant) is split into seed chunks; one chunk is one child
process on one GPU (run_hpo_copies.py --run-trial, so the factored copies patches apply),
with its own CSV. Resuming: rerun the same command, only missing seeds are trained.

Usage (server, in cc/ with the venv active):
  python run_reg_check.py --smoke                 # 1 seed, 15 epochs, results/reg_check_smoke/
  python run_reg_check.py --max-parallel 4
  python run_reg_check.py --collect               # summary.csv only

Results (results/reg_check/):
  <dataset>/<trial>_<variant>/seeds_<a>-<b>.csv  <- one row per seed
  <dataset>/logs/, jobs/
  summary.csv      <- per dataset, finalist and variant: val / test / train mean +- std
  selection.csv    <- per dataset and variant: the finalist picked by val, and val-test correlation
"""

# Imports
import run_hpo_copies  # noqa: F401 - the children run it; checks it is there
from train08 import EPOCHS
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

SOURCE = join("results", "hpo_copies")
RESULTS = join("results", "reg_check")
DATASETS = ["em_user", "ppi_bp"]  # longest first
FINALISTS = 3
SEEDS = list(range(10))
CHUNK = 5  # seeds per child
VARIANTS = ["drop", "wd", "both"]


def finalist_trials(name):
    """The study's FINALISTS best complete trials (OOM ones score 0), as run_hpo picks them."""
    t = pd.read_csv(join(SOURCE, "trials.csv"))
    t = t[(t["dataset"] == name) & (t["setup"] == "colors") & (t["state"] == "COMPLETE") & (t["value"] > 0)]
    t = t.sort_values(["value", "trial"], ascending=[False, True])
    return [int(n) for n in t["trial"].head(FINALISTS)]


def base_hps(name, trial):
    with open(join(SOURCE, name, "colors", "jobs", f"{trial}_final.json")) as f:
        return json.load(f)["hps"]


def regularize(hps, variant):
    hps = copy.deepcopy(hps)
    if variant in ("drop", "both"):
        for k in ("gnn_dropout_rate", "msg_dropout", "input_dropout"):
            hps[k] = round(min(0.8, hps[k] + 0.2), 2)
    if variant in ("wd", "both"):
        hps["weight_decay"] = max(1e-3, 10 * hps["weight_decay"])
    return hps


def plan(args, root):
    """[(seconds estimate, dataset, label, seeds, job_path, log_path)] for every chunk with missing seeds."""
    seeds = SEEDS[:1] if args.smoke else SEEDS
    trials = pd.read_csv(join(SOURCE, "trials.csv"))
    jobs = []
    for name in DATASETS:
        d = join(root, name)
        os.makedirs(join(d, "jobs"), exist_ok=True)
        os.makedirs(join(d, "logs"), exist_ok=True)
        for trial in finalist_trials(name):
            secs = float(trials[(trials["dataset"] == name) & (trials["trial"] == trial)]["seconds_mean"].iloc[0])
            for variant in VARIANTS:
                hps = regularize(base_hps(name, trial), variant)
                label = f"{trial}_{variant}"
                os.makedirs(join(d, label), exist_ok=True)
                for i in range(0, len(seeds), CHUNK):
                    part = seeds[i:i + CHUNK]
                    tag = f"seeds_{part[0]}-{part[-1]}"
                    csv = join(d, label, f"{tag}.csv")
                    if os.path.exists(csv):
                        df = pd.read_csv(csv, on_bad_lines="skip").dropna(subset=["seed", "val_f1"])
                        if set(part) <= set(df["seed"].astype(int)):
                            continue
                    job_path = join(d, "jobs", f"{label}_{tag}.json")
                    with open(job_path, "w") as f:
                        json.dump({"dataset": name, "number": label, "hps": hps, "seeds": part, "csv": csv,
                                   "prune_below": None}, f)
                    jobs.append((secs * len(part), name, label, part, job_path,
                                 join(d, "logs", f"{label}_{tag}.log")))
    return sorted(jobs, key=lambda j: -j[0])  # longest chunks first


def run_all(args, root, epochs):
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
            _, name, label, part, job_path, log_path = jobs.pop(0)
            env = dict(os.environ)
            if slots[slot] is not None:
                env["CUDA_VISIBLE_DEVICES"] = slots[slot]
            log = open(log_path, "a")
            cmd = [sys.executable, os.path.abspath(run_hpo_copies.__file__), "--run-trial", job_path,
                   "--data-root", args.data_root, "--epochs", str(epochs)]
            running[slot] = (subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log,
                             f"{name} {label} seeds {part[0]}-{part[-1]}")
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
    for name in DATASETS:
        for trial in finalist_trials(name):
            sources = {"base": [join(SOURCE, name, "colors", "trials", f"{trial}.csv")]}
            for variant in VARIANTS:
                d = join(root, name, f"{trial}_{variant}")
                sources[variant] = [join(d, f) for f in sorted(os.listdir(d))] if os.path.isdir(d) else []
            for variant, paths in sources.items():
                df = read_seeds(paths)
                if df is None or df.empty:
                    continue
                oom = bool((df["val_f1"] < 0).any())
                rows.append({"dataset": name, "trial": trial, "variant": variant, "n_seeds": len(df), "oom": oom,
                             **{f"{k}_{s}": round(100 * getattr(df[f"{k}_f1"], s)(), 2)
                                for k in ("val", "test", "train") for s in ("mean", "std")}})
    if not rows:
        return
    summary = pd.DataFrame(rows)
    summary.to_csv(join(root, "summary.csv"), index=False)
    print(summary.to_string(index=False))

    # Selection: per variant, the finalist with the best val (as run_hpo picks), and how val tracks test
    sel = []
    ok = summary[~summary["oom"]]
    for (name, variant), g in ok.groupby(["dataset", "variant"], sort=False):
        best = g.sort_values(["val_mean", "trial"], ascending=[False, True]).iloc[0]
        sel.append({"dataset": name, "variant": variant, "picked_trial": int(best["trial"]),
                    "val": best["val_mean"], "test": best["test_mean"], "test_std": best["test_std"],
                    "train": best["train_mean"], "best_test_of_3": g["test_mean"].max()})
    for name, g in ok.groupby("dataset", sort=False):
        if len(g) >= 3:
            sel.append({"dataset": name, "variant": "all (val-test corr)",
                        "val": round(g["val_mean"].corr(g["test_mean"]), 2)})
    selection = pd.DataFrame(sel)
    selection.to_csv(join(root, "selection.csv"), index=False)
    print(selection.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="em_user and ppi_bp finalists with stronger regularization.")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--data-root", default=join("data", "raw"))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--smoke", action="store_true", help="1 seed, 15 epochs, into results/reg_check_smoke/")
    parser.add_argument("--collect", action="store_true", help="Only rebuild summary.csv and selection.csv")
    args = parser.parse_args()
    root = RESULTS + ("_smoke" if args.smoke else "")
    if args.collect:
        collect(root)
    else:
        run_all(args, root, 15 if args.smoke else args.epochs)


if __name__ == "__main__":
    main()
