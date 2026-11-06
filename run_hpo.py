"""
Yuli Tshuva.
run_hpo.py - our one configuration, tuned by an Optuna search per dataset, then the
best trials on 10 seeds, for one results table against GLASS and GPEN.

The configuration ("colors", GLASS's training recipe, GLASS's splits): inside / outside
edges as two colors. On the real datasets without copies (Gin_Gout: their copies
graphs do not fit in GPU memory); on the synthetic ones with copies (I_O: every shared
node replaced by one copy per subgraph). "plain" (the graph as is, coloring G) is kept
as an optional baseline (--setups plain), off by default.

Search space (SPACE): aggregation, pooling, hidden size, layers, lr, message dropout,
input dropout, batch size (x0.5 / x1 / x2 GLASS's), weight decay, norm, jumping
knowledge, head (linear or 64-unit MLP) and, on the synthetic datasets, the edge-count
features. Everything else is run_edge_colors' GLASS training.

A trial is 3 seeds (0-2), scored by their mean validation micro-F1. Median pruning:
once a study has MIN_FOR_PRUNING complete trials, a trial whose seed-0 validation is
below their seed-0 median stops there. TPE (multivariate, constant liar for the trials
running at once). When a study has its trials, its FINALISTS best trials run seeds
3-9; final.csv keeps, per dataset and setup, the finalist with the best validation over
all 10 seeds, with its test mean +- std. Test is never used for selection.

Processes: this one owns every study (journal file, single writer) and asks for the
trials; each trial runs in a child process on one GPU slot and writes one CSV row per
seed. Resuming: rerun the same command. A trial left running by a kill is relaunched
with the same hps, and only its missing seeds are trained.

Usage (server):
  python run_hpo.py --max-parallel 4
  python run_hpo.py --max-parallel 8 --jobs-per-gpu 2
  python run_hpo.py --dataset density --smoke     # 3 trials of 1 seed, 15 epochs, results/hpo_smoke/
  python run_hpo.py --collect                     # rebuild final.csv / trials.csv only

Results (results/hpo/):
  <dataset>/<setup>/study.log        <- the Optuna journal
  <dataset>/<setup>/trials/<n>.csv   <- one row per seed of trial n
  <dataset>/<setup>/logs/<n>.log
  trials.csv                         <- every trial: params, state, val/test means
  final.csv                          <- per dataset and setup: the chosen trial on 10 seeds
"""

# Imports
from run_gpen_compare import (DATASETS, REAL, GLASS_CONFIGS, load_dataset, prepare, glass_split,
                              build_colored_relations, build_copy_edges, build_member_pairs, train,
                              make_row, append_row, is_oom, free_memory)
from run_edge_colors import COLORINGS, AGGREGATIONS, POOLINGS, config_hps, config_key, uses_copies
from train08 import EPOCHS
from os.path import join
import argparse
import json
import optuna
from optuna.distributions import CategoricalDistribution as Cat, FloatDistribution, IntDistribution
import os
import pandas as pd
import subprocess
import sys
import time
import torch

SETUPS = {"plain": {"real": "G", "synthetic": "G"},
          "colors": {"real": "Gin_Gout", "synthetic": "I_O"}}
# Trials per study; a hpo_neuro seed takes ~11 minutes at GLASS's size
TRIALS = {"hpo_neuro": 40, "hpo_metab": 80, "em_user": 80}
DEFAULT_TRIALS = 150
TRIAL_SEEDS, FINAL_SEEDS, FINALISTS = 3, 10, 3
MIN_FOR_PRUNING = 10
# Longest seeds first, so the slowest studies start first
ORDER = ["hpo_neuro", "hpo_metab", "em_user", "ppi_bp", "coreness", "density", "component", "cut_ratio"]
RESULTS = join("results", "hpo")
SMOKE = {"trials": 3, "trial_seeds": 1, "finalists": 1, "final_seeds": 2, "epochs": 15}


def space(name):
    real = name in REAL
    d = {"aggregation": Cat(AGGREGATIONS),
         "pooling": Cat(POOLINGS),
         "hidden": Cat([32, 64, 128, 256] if real else [8, 16, 32, 64, 128]),
         "layers": IntDistribution(1, 3),
         "lr": FloatDistribution(1e-4, 1e-2, log=True),
         "dropout": FloatDistribution(0.0, 0.6, step=0.1),
         "input_dropout": FloatDistribution(0.0, 0.6, step=0.1),
         "batch": Cat([0.5, 1.0, 2.0]),
         "weight_decay": Cat([0.0, 1e-5, 1e-4, 1e-3]),
         "norm": Cat(["graph", "batch", "layer", "none"]),
         "jk": Cat([True, False]),
         "head": Cat([0, 64])}
    if not real:
        d["counts"] = Cat([False, True])
    return d


def coloring_of(name, setup):
    return SETUPS[setup]["real" if name in REAL else "synthetic"]


def trial_hps(name, setup, p):
    """The full hps of a trial: run_edge_colors' GLASS-trained config, with the sampled params on top."""
    coloring = coloring_of(name, setup)
    hps = config_hps(name, config_key(coloring, "glass", p["aggregation"], p["pooling"]))
    hidden, layers = p["hidden"], p["layers"]
    width = hidden * layers if p["jk"] else hidden
    hps.update({"gnn_embedding_dim": hidden, "gnn_hidden_dim": hidden, "k_gnn_layers": layers, "jk": p["jk"],
                "pma_heads": next(h for h in (4, 2, 1) if width % h == 0),
                "lr": p["lr"], "gnn_dropout_rate": p["dropout"], "msg_dropout": p["dropout"],
                "input_dropout": p["input_dropout"],
                "batch_size": max(1, round(GLASS_CONFIGS[name]["batch_size"] * p["batch"])),
                "weight_decay": p["weight_decay"], "norm": None if p["norm"] == "none" else p["norm"],
                "color_mlp_hidden_dims": p["head"]})
    if p.get("counts"):
        hps["count_skip"] = True
        hps["copy_counts"] = uses_copies(COLORINGS[coloring])
    return hps


# ---------------------------------------------------------------------------
# Studies
# ---------------------------------------------------------------------------

def study_dir(root, name, setup):
    return join(root, name, setup)


def make_storage(path):
    # One process writes the journal, so the plain open-file lock is enough (no symlinks)
    try:
        from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
        return optuna.storages.JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))
    except ImportError:  # optuna < 4
        return optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(
            path, lock_obj=optuna.storages.JournalFileOpenLock(path)))


def load_study(root, name, setup, n_trials):
    d = study_dir(root, name, setup)
    os.makedirs(join(d, "trials"), exist_ok=True)
    sampler = optuna.samplers.TPESampler(seed=0, multivariate=True, constant_liar=True,
                                         n_startup_trials=max(5, min(20, n_trials // 6)))
    return optuna.create_study(direction="maximize", sampler=sampler, study_name=f"{name}_{setup}",
                               storage=make_storage(join(d, "study.log")), load_if_exists=True)


def trial_csv(root, name, setup, number):
    return join(study_dir(root, name, setup), "trials", f"{number}.csv")


def read_trial(root, name, setup, number):
    """The trial's seeds (a row cut off mid-write is dropped), or None."""
    path = trial_csv(root, name, setup, number)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, on_bad_lines="skip").dropna(subset=["seed", "val_f1", "test_f1"])
    df["seed"] = df["seed"].astype(int)
    return df.drop_duplicates("seed", keep="last")


def prune_threshold(study):
    """Median seed-0 validation of the finished trials (pruned ones too, or the bar would only rise);
    None before MIN_FOR_PRUNING of them."""
    states = [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED]
    vals = [t.user_attrs["seed0_val"] for t in study.get_trials(deepcopy=False, states=states)
            if "seed0_val" in t.user_attrs and not t.user_attrs.get("oom")]
    return float(pd.Series(vals).median()) if len(vals) >= MIN_FOR_PRUNING else None


def finish_trial(study, root, name, setup, trial, n_seeds, crashed):
    """Tell the study a trial's outcome from its CSV, once its child exited."""
    df = read_trial(root, name, setup, trial.number)
    live = optuna.trial.Trial(study, trial._trial_id)
    if crashed or df is None or df.empty:
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
        return "failed"
    seed0 = df[df["seed"] == 0]["val_f1"]
    if (df["val_f1"] < 0).any():
        live.set_user_attr("oom", True)
        study.tell(trial.number, 0.0)  # ranks last, and tells TPE to stay away
        return "OOM"
    if len(seed0):
        live.set_user_attr("seed0_val", float(seed0.iloc[0]))
    if set(range(n_seeds)) <= set(df["seed"]):
        value = float(df[df["seed"] < n_seeds]["val_f1"].mean())
        study.tell(trial.number, value)
        return f"val {value:.4f}"
    if len(seed0):
        # Stopped after seed 0 by the median rule
        live.report(float(seed0.iloc[0]), 0)
        study.tell(trial.number, state=optuna.trial.TrialState.PRUNED)
        return f"pruned (seed 0 val {float(seed0.iloc[0]):.4f})"
    study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    return "failed"


def finalists(study, k):
    done = [t for t in study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
            if not t.user_attrs.get("oom")]
    return sorted(done, key=lambda t: (-t.value, t.number))[:k]


# ---------------------------------------------------------------------------
# One trial (child process)
# ---------------------------------------------------------------------------

def run_trial(job_path, data_root, epochs):
    with open(job_path) as f:
        job = json.load(f)
    name, hps, path = job["dataset"], job["hps"], job["csv"]
    done = set()
    if os.path.exists(path):
        df = pd.read_csv(path, on_bad_lines="skip").dropna(subset=["seed", "val_f1"])
        done = set(df["seed"].astype(int))
        with open(path, "rb+") as f:  # end a row cut off mid-write
            f.seek(0, os.SEEK_END)
            if f.tell() > 0:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    f.write(b"\n")
    todo = [s for s in job["seeds"] if s not in done]
    if not todo:
        return
    start = time.time()
    prep = prepare(load_dataset(name, data_root))
    prep["name"] = name
    if hps.get("copies") or hps.get("count_skip"):
        prep["copy_edges"] = build_copy_edges(prep["subgraphs"], prep["graph_edges"], prep["n_nodes"])
    build_colored_relations(prep, hps)
    if hps["readout"] in ("pma", "transformer") and hps.get("pma_sabs", 0) > 0:
        prep["member_pairs"] = build_member_pairs(prep["member_sub"])
    print(f"[{name}] trial {job['number']}: loaded in {time.time() - start:.1f}s, seeds {todo}\n{hps}", flush=True)
    for i, seed in enumerate(todo):
        start = time.time()
        try:
            res = train(glass_split(prep, name, seed), hps, epochs=epochs, seed=seed)
        except RuntimeError as e:
            if not is_oom(e):
                raise
            print(f"[{name}] OOM on seed {seed}", flush=True)
            append_row(path, make_row(name, hps, job["number"], seed,
                                      {"train_f1": -1.0, "val_f1": -1.0, "test_f1": -1.0, "epochs": 0}, 0.0))
            return
        append_row(path, make_row(name, hps, job["number"], seed, res, time.time() - start))
        free_memory()
        print(f"[{name}] trial {job['number']} seed {seed}: val {res['val_f1']:.4f}, test {res['test_f1']:.4f} "
              f"({res['epochs']} epochs, {time.time() - start:.0f}s)", flush=True)
        threshold = job.get("prune_below")
        if seed == 0 and threshold is not None and res["val_f1"] < threshold:
            print(f"[{name}] pruned: seed 0 val below the median {threshold:.4f}", flush=True)
            return


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

def run_all(args, root, cfg):
    names = [n for n in ORDER if n not in args.exclude] if args.dataset == "all" else [args.dataset]
    setups = args.setups
    studies = {(n, s): load_study(root, n, s, cfg["trials"](n)) for n in names for s in setups}
    n_gpus = torch.cuda.device_count()
    slots = [str(g) for _ in range(args.jobs_per_gpu) for g in range(n_gpus)][:args.max_parallel] or [None]
    running = {}  # slot -> (proc, log, kind, name, setup, trial)
    failed_children, failed_finals = 0, set()  # a finalist whose seeds crashed is not retried this session

    # Trials left RUNNING by a kill: queued again with their own hps
    orphans = [(n, s, t) for (n, s), st in studies.items()
               for t in st.get_trials(deepcopy=False, states=[optuna.trial.TrialState.RUNNING])]
    print(f"{len(slots)} at a time on {n_gpus} GPU(s); {len(orphans)} trial(s) to resume; studies {list(studies)}")

    def launch(slot, kind, name, setup, trial, seeds, prune_below):
        d = study_dir(root, name, setup)
        job = {"dataset": name, "number": trial.number, "hps": trial.user_attrs["hps"], "seeds": seeds,
               "csv": trial_csv(root, name, setup, trial.number), "prune_below": prune_below}
        os.makedirs(join(d, "jobs"), exist_ok=True)
        os.makedirs(join(d, "logs"), exist_ok=True)
        job_path = join(d, "jobs", f"{trial.number}_{kind}.json")
        with open(job_path, "w") as f:
            json.dump(job, f)
        env = dict(os.environ)
        if slots[slot] is not None:
            env["CUDA_VISIBLE_DEVICES"] = slots[slot]
        log = open(join(d, "logs", f"{trial.number}.log"), "a")
        cmd = [sys.executable, os.path.abspath(__file__), "--run-trial", job_path, "--data-root", args.data_root,
               "--epochs", str(cfg["epochs"])]
        running[slot] = (subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log, kind, name,
                         setup, trial)
        print(f"  {kind} {name}/{setup} trial {trial.number} on GPU {slots[slot]} "
              f"(seeds {seeds[0]}-{seeds[-1]}{'' if prune_below is None else f', prune below {prune_below:.4f}'})",
              flush=True)

    def busy(name, setup):
        return [r for r in running.values() if r[3] == name and r[4] == setup]

    def next_job():
        """(kind, name, setup, trial, seeds, prune_below) of the next job, or None."""
        while orphans:
            n, s, t = orphans.pop(0)
            return "trial", n, s, t, list(range(cfg["trial_seeds"])), None
        for (n, s), st in studies.items():
            states = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED,
                      optuna.trial.TrialState.RUNNING)
            if len(st.get_trials(deepcopy=False, states=states)) < cfg["trials"](n):
                trial = st.ask(space(n))
                params = trial.params
                trial.set_user_attr("hps", trial_hps(n, s, params))
                return ("trial", n, s, st.get_trials(deepcopy=False)[trial.number], list(range(cfg["trial_seeds"])),
                        prune_threshold(st))
        for (n, s), st in studies.items():
            if busy(n, s) and any(r[2] == "trial" for r in busy(n, s)):
                continue  # finalists are known once every trial is told
            active = {r[5].number for r in busy(n, s)}
            for t in finalists(st, cfg["finalists"]):
                df = read_trial(root, n, s, t.number)
                have = set() if df is None else set(df["seed"])
                if t.number not in active and (n, s, t.number) not in failed_finals and not set(range(cfg["final_seeds"])) <= have \
                        and not (df is not None and (df["val_f1"] < 0).any()):
                    return "final", n, s, t, list(range(cfg["final_seeds"])), None
        return None

    while True:
        for slot in [i for i in range(len(slots)) if i not in running]:
            job = next_job()
            if job is None:
                break
            launch(slot, *job)
        if not running:
            break
        time.sleep(5)
        for slot, (proc, log, kind, name, setup, trial) in list(running.items()):
            if proc.poll() is None:
                continue
            log.close()
            del running[slot]
            if proc.returncode != 0:
                failed_children += 1
                print(f"  FAILED {name}/{setup} trial {trial.number} (exit code {proc.returncode}) - see its log")
            if kind == "trial":
                outcome = finish_trial(studies[(name, setup)], root, name, setup, trial, cfg["trial_seeds"],
                                       proc.returncode != 0)
                print(f"  done {name}/{setup} trial {trial.number}: {outcome}", flush=True)
            elif proc.returncode != 0:
                failed_finals.add((name, setup, trial.number))
            else:
                print(f"  done final seeds of {name}/{setup} trial {trial.number}", flush=True)
            if failed_children >= 20:
                print("20 children failed - stopping; fix the cause and rerun the same command")
                for r in running.values():
                    r[0].wait()
                collect(root, cfg)
                return
    collect(root, cfg)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def collect(root, cfg):
    trial_rows, final_rows = [], []
    for name in ORDER:
        for setup in SETUPS:
            if not os.path.exists(join(study_dir(root, name, setup), "study.log")):
                continue
            study = load_study(root, name, setup, cfg["trials"](name))
            for t in study.get_trials(deepcopy=False):
                df = read_trial(root, name, setup, t.number)
                first = None if df is None else df[df["seed"] < cfg["trial_seeds"]]
                trial_rows.append({"dataset": name, "setup": setup, "trial": t.number, "state": t.state.name,
                                   "value": t.value, **t.params,
                                   "val_mean": None if first is None or first.empty else round(100 * first["val_f1"].mean(), 2),
                                   "test_mean": None if first is None or first.empty else round(100 * first["test_f1"].mean(), 2),
                                   "seconds_mean": None if first is None or first.empty else round(first["seconds"].mean(), 1)})
            best = None
            for t in finalists(study, cfg["finalists"]):
                df = read_trial(root, name, setup, t.number)
                if df is None or not set(range(cfg["final_seeds"])) <= set(df["seed"]):
                    continue
                df = df[df["seed"] < cfg["final_seeds"]]
                row = {"dataset": name, "setup": setup, "coloring": coloring_of(name, setup), "trial": t.number,
                       "n_seeds": len(df), "val_mean": round(100 * df["val_f1"].mean(), 2),
                       "val_std": round(100 * df["val_f1"].std(), 2),
                       "test_mean": round(100 * df["test_f1"].mean(), 2), "test_std": round(100 * df["test_f1"].std(), 2),
                       "train_mean": round(100 * df["train_f1"].mean(), 2), "seconds_mean": round(df["seconds"].mean(), 1),
                       **t.params}
                if best is None or (row["val_mean"], -row["trial"]) > (best["val_mean"], -best["trial"]):
                    best = row
            if best:
                final_rows.append(best)
    if trial_rows:
        pd.DataFrame(trial_rows).to_csv(join(root, "trials.csv"), index=False)
    if final_rows:
        final = pd.DataFrame(final_rows)
        final.to_csv(join(root, "final.csv"), index=False)
        print(final[["dataset", "setup", "trial", "val_mean", "test_mean", "test_std"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Optuna search of our two setups on GPEN's eight benchmarks.")
    parser.add_argument("--dataset", default="all", choices=DATASETS + ["all"])
    parser.add_argument("--exclude", nargs="+", default=[], choices=DATASETS)
    parser.add_argument("--setups", nargs="+", default=["colors"], choices=list(SETUPS))
    parser.add_argument("--max-parallel", type=int, default=4, help="Trials at once; each holds a dataset in RAM")
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--trials-scale", type=float, default=1.0, help="Multiplies every study's trial count")
    parser.add_argument("--data-root", default=join("data", "raw"))
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Epoch cap; GLASS's early stopping decides")
    parser.add_argument("--smoke", action="store_true", help="3 trials of 1 seed, 15 epochs, into results/hpo_smoke/")
    parser.add_argument("--collect", action="store_true", help="Only rebuild trials.csv and final.csv")
    parser.add_argument("--run-trial", help=argparse.SUPPRESS)  # internal: one trial in this process
    args = parser.parse_args()

    if args.run_trial:
        run_trial(args.run_trial, args.data_root, args.epochs)
        return
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    root = RESULTS + ("_smoke" if args.smoke else "")
    if args.smoke:
        cfg = {"trials": lambda n: SMOKE["trials"], "trial_seeds": SMOKE["trial_seeds"],
               "finalists": SMOKE["finalists"], "final_seeds": SMOKE["final_seeds"], "epochs": SMOKE["epochs"]}
    else:
        cfg = {"trials": lambda n: max(1, round(TRIALS.get(n, DEFAULT_TRIALS) * args.trials_scale)),
               "trial_seeds": TRIAL_SEEDS, "finalists": FINALISTS, "final_seeds": FINAL_SEEDS,
               "epochs": args.epochs}
    if args.collect:
        collect(root, cfg)
    else:
        run_all(args, root, cfg)


if __name__ == "__main__":
    main()
