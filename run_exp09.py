"""
Yuli Tshuva.
run_exp09.py - Driver for the learned-pooling comparison (Channel 1 of the
post-exp08 proposals). Runs the five new readouts plus the exp08 control,
puts error bars on each, and writes the summary tables.

Three stages, run in order by default:
  studies : one Optuna study per readout (one job per GPU; when a GPU frees up,
            the next job starts). Selection on validation AUC only.
  seeds   : retrain each study's best config over --seeds seeds, so the test
            AUCs come with error bars.
  report  : aggregate everything into results/exp09/<dataset>/summary.{md,tex}
            and print it.

Usage (after connecting, e.g.
  srun -p killable --gres=gpu:a5000:2 -c 50 --mem=16G -t 24:00:00 --pty bash
and activating the venv):
  python run_exp09.py                          # all three stages, rgat backbone
  python run_exp09.py --backbones rgat rgcn    # both relational backbones
  python run_exp09.py --stage report           # just rebuild the tables
  python run_exp09.py --trials 5 --seeds 2     # short end-to-end rehearsal

Progress: tail -f logs/<job name>.log
Interrupted study jobs are cheap to rerun: already-tried hp configs are skipped
via results/exp09/<dataset>/<backbone>_<readout>/hp_results.json.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from os.path import join

import pandas as pd

READOUTS = ["mean", "node_score_mean", "multistat", "gated_attn", "supernode", "pma"]

# Row descriptions for the summary table, in the style of the exp08 report.
READOUT_DESC = {
    "mean": "Control (exp08). Parameter-free mean of the member node embeddings, then a color MLP",
    "node_score_mean": "Baseline. Color score is the mean of the node head's sigmoid scores; no color head",
    "multistat": "1.3 Multi-statistic readout: mean || max || sum concatenated before the color MLP",
    "gated_attn": "1.1 Gated attention (MIL) pooling; scores softmax-normalized within each color",
    "supernode": "1.2 One virtual node per color as a third relation; its embedding feeds the color head",
    "pma": "1.4 Pooling by Multihead Attention, optionally after self-attention among the members",
}


def n_gpus():
    try:
        import torch
        return max(1, torch.cuda.device_count())
    except Exception:
        return 1


def run_jobs(jobs):
    """Launch (name, cmd) pairs one per GPU, starting the next as a GPU frees up."""
    os.makedirs("logs", exist_ok=True)
    gpus = n_gpus()
    print(f"Found {gpus} GPU(s). Running {len(jobs)} jobs, one per GPU at a time.", flush=True)

    queue = list(jobs)
    running = {}  # gpu -> (process, job_name, log_file)
    failed = []

    while queue or running:
        for gpu in range(gpus):
            if gpu not in running and queue:
                name, cmd = queue.pop(0)
                log = open(join("logs", f"{name}.log"), "w")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
                running[gpu] = (proc, name, log)
                print(f"Launched {name} on GPU {gpu} (pid {proc.pid})", flush=True)

        for gpu in list(running):
            proc, name, log = running[gpu]
            if proc.poll() is not None:
                log.close()
                status = "done" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
                print(f"{name}: {status}", flush=True)
                if proc.returncode != 0:
                    failed.append(name)
                del running[gpu]

        time.sleep(10)

    return failed


def study_jobs(args):
    return [(f"exp09_{b}_{r}",
             [sys.executable, "train09.py", "--readout", r, "--backbone", b,
              "--node-input", args.node_input,
              "--dataset", args.dataset, "--trials", str(args.trials),
              "--epochs", str(args.epochs)])
            for b in args.backbones for r in args.readouts]


def seed_jobs(args):
    return [(f"exp09_{b}_{r}_seeds",
             [sys.executable, "train09.py", "--readout", r, "--backbone", b,
              "--node-input", args.node_input,
              "--dataset", args.dataset, "--seed-eval", "--seeds", str(args.seeds),
              "--epochs", str(args.epochs)])
            for b in args.backbones for r in args.readouts]


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def _mean_over_datasets(df, group_cols):
    """
    Average over the datasets of a run before aggregating. For pregnancy this is
    a no-op (one dataset); for the disease sweep it reproduces the exp08
    convention of reporting the mean over the eight datasets.
    """
    return df.groupby(group_cols, as_index=False).mean(numeric_only=True)


def collect(dataset, backbones, readouts):
    """
    Build one summary row per (backbone, readout): the best configuration by
    validation AUC, and its test AUC over seeds.
    """
    rows = []
    for backbone in backbones:
        for readout in readouts:
            run_dir = join("results", "exp09", dataset, f"{backbone}_{readout}")
            trial_path = join(run_dir, "trial_results.csv")
            if not os.path.exists(trial_path):
                print(f"  (skipping {backbone}/{readout}: no study results yet)")
                continue

            trials = pd.read_csv(trial_path)
            trials["val_score"] = (trials["val_gnn_auc"] + trials["val_color_auc"]) / 2
            per_config = _mean_over_datasets(trials, ["hps"])
            best = per_config.loc[per_config["val_score"].idxmax()]

            row = {
                "backbone": backbone,
                "readout": readout,
                "description": READOUT_DESC.get(readout, ""),
                "n_configs": trials["hps"].nunique(),
                "val_node": best["val_gnn_auc"],
                "val_color": best["val_color_auc"],
                "test_node": best["test_gnn_auc"],
                "test_color": best["test_color_auc"],
                "test_node_std": float("nan"),
                "test_color_std": float("nan"),
                "n_seeds": 0,
                "best_hps": best["hps"],
            }

            # Prefer the multi-seed test AUCs when the seed stage has been run
            seed_path = join("results", "exp09", dataset, "seed_results.csv")
            if os.path.exists(seed_path):
                seeds = pd.read_csv(seed_path)
                seeds = seeds[(seeds["backbone"] == backbone) & (seeds["readout"] == readout)]
                # Only trust seed rows that used the config we are reporting
                seeds = seeds[seeds["hps"] == best["hps"]]
                if len(seeds):
                    per_seed = _mean_over_datasets(seeds, ["trial"])
                    row.update(test_node=per_seed["test_gnn_auc"].mean(),
                               test_color=per_seed["test_color_auc"].mean(),
                               test_node_std=per_seed["test_gnn_auc"].std(ddof=0),
                               test_color_std=per_seed["test_color_auc"].std(ddof=0),
                               n_seeds=len(per_seed))

            rows.append(row)

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("test_color", ascending=False).reset_index(drop=True)
    return df


def _pm(mean, std, seeds):
    return f"{mean:.3f} ± {std:.3f}" if seeds > 1 else f"{mean:.3f}"


def to_markdown(df, dataset, seeds_note):
    lines = [f"# Experiment 09 - learned pooling ({dataset})", "",
             "Backbone frozen at the exp08 winner (two-relation graph, MiniLM features, no color",
             "embedding). Configurations selected on validation AUC only; test AUC is never",
             f"optimized on. {seeds_note}", "",
             "| Backbone | Readout | What was done | Configs | Val AUC (node/color) | Test AUC node | Test AUC color |",
             "|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['backbone']} | `{r['readout']}` | {r['description']} | {r['n_configs']} | "
            f"{r['val_node']:.3f} / {r['val_color']:.3f} | "
            f"{_pm(r['test_node'], r['test_node_std'], r['n_seeds'])} | "
            f"{_pm(r['test_color'], r['test_color_std'], r['n_seeds'])} |")

    lines += ["", "## Best configuration per readout", ""]
    for _, r in df.iterrows():
        lines.append(f"- **{r['backbone']}/{r['readout']}**: `{r['best_hps']}`")
    return "\n".join(lines) + "\n"


def to_latex(df, dataset, seeds_note):
    """A booktabs table in the format of the exp08 report, ready to \\input."""
    esc = lambda s: str(s).replace("_", r"\_")
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\begin{tabularx}{\textwidth}{llXcc}", r"\toprule",
             r"Backbone & Readout & What was done & \makecell{Val AUC\\(node/color)} "
             r"& \makecell{Test AUC\\node\\color} \\", r"\midrule"]
    for _, r in df.iterrows():
        node = _pm(r["test_node"], r["test_node_std"], r["n_seeds"]).replace("±", r"$\pm$")
        color = _pm(r["test_color"], r["test_color_std"], r["n_seeds"]).replace("±", r"$\pm$")
        lines.append(
            f"{esc(r['backbone'])} & \\texttt{{{esc(r['readout'])}}} & {esc(r['description'])} & "
            f"{r['val_node']:.3f} / {r['val_color']:.3f} & \\makecell{{{node}\\\\{color}}} \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}",
              rf"\caption{{\texttt{{{esc(dataset)}}}. Best configuration per readout, selected by "
              rf"validation AUC. {esc(seeds_note)}}}",
              r"\label{tab:exp09-pooling}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def report(dataset, backbones, readouts):
    print(f"\nBuilding the summary for {dataset}...")
    df = collect(dataset, backbones, readouts)
    if df.empty:
        print("No results found under results/exp09 - run the studies first.")
        return

    n_seeds = int(df["n_seeds"].max())
    seeds_note = (f"Test AUC is the mean ± standard deviation over {n_seeds} seeds "
                  f"(only model init and dropout vary; the split is fixed)."
                  if n_seeds > 1 else
                  "Test AUC is from the single selection run - run the seed stage for error bars.")

    out_dir = join("results", "exp09", dataset)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(join(out_dir, "summary.csv"), index=False)
    md = to_markdown(df, dataset, seeds_note)
    with open(join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(join(out_dir, "summary.tex"), "w", encoding="utf-8") as f:
        f.write(to_latex(df, dataset, seeds_note))

    print("\n" + md)

    control = df[df["readout"] == "mean"]
    if len(control):
        base_node, base_color = control.iloc[0]["test_node"], control.iloc[0]["test_color"]
        print("Change vs. the exp08 control (mean pooling), in AUC points:")
        for _, r in df.iterrows():
            if r["readout"] == "mean":
                continue
            print(f"  {r['backbone']}/{r['readout']:<16s} "
                  f"node {100 * (r['test_node'] - base_node):+.1f}   "
                  f"color {100 * (r['test_color'] - base_color):+.1f}")

    print(f"\nWritten: {join(out_dir, 'summary.md')}, summary.tex, summary.csv")


def results_tag(dataset, node_input):
    """Mirrors train08.results_tag, without importing torch into the driver."""
    return dataset if node_input == "feat" else f"{dataset}-{node_input}"


def main():
    parser = argparse.ArgumentParser(description="Run and report the exp09 learned-pooling comparison.")
    parser.add_argument("--dataset", type=str, default="pregnancy",
                        help="diseases, pregnancy, or another data/ folder with the pregnancy "
                             "layout (e.g. pregnancy_C for the anti-collapse embeddings)")
    parser.add_argument("--backbones", type=str, nargs="+", default=["rgat"], choices=["rgat", "rgcn"])
    parser.add_argument("--readouts", type=str, nargs="+", default=READOUTS, choices=READOUTS)
    parser.add_argument("--trials", type=int, default=60, help="Optuna trials per readout")
    parser.add_argument("--node-input", type=str, default="feat",
                        choices=["feat", "const", "random", "learned"],
                        help="What the GNN reads at its input layer, passed through to train09.py. "
                             "Anything but 'feat' drops the embeddings from the input (the graph "
                             "is still built from them) and writes to results/exp09/<dataset>-<node input>/")
    parser.add_argument("--seeds", type=int, default=5, help="Seeds per best configuration")
    parser.add_argument("--epochs", type=int, default=10000,
                        help="Epoch cap passed to train09.py. The default relies on early stopping.")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "studies", "seeds", "report"])
    args = parser.parse_args()

    failed = []
    if args.stage in ("all", "studies"):
        print("=== Stage 1/3: hyperparameter studies ===")
        failed += run_jobs(study_jobs(args))

    if args.stage in ("all", "seeds"):
        if failed:
            print(f"Studies failed: {failed} - fix them before the seed stage, "
                  f"otherwise the best configs are selected from partial results.")
            return
        print("\n=== Stage 2/3: multi-seed evaluation ===")
        failed += run_jobs(seed_jobs(args))

    if args.stage in ("all", "report"):
        print("\n=== Stage 3/3: report ===")
        report(results_tag(args.dataset, args.node_input), args.backbones, args.readouts)

    if failed:
        print(f"\nFailed jobs: {failed} - check logs/<name>.log and rerun.")
    else:
        print("\nAll done. Export the results/exp09/ folder.")


if __name__ == "__main__":
    main()
