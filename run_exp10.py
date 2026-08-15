"""
Yuli Tshuva.
run_exp10.py - Driver for the graph-construction comparison (Channel 2 of the
post-exp08 proposals). Sweeps the similarity threshold of the hybrid graph
against the fixed k=20 kNN control, puts error bars on each, and writes the
summary tables.

Four stages, run in order by default:
  candidates : precompute the k_max nearest neighbours of every node and cache
               them. Runs serially and first, on purpose - the study jobs all
               read this cache and would otherwise race to build it.
  studies    : one Optuna study per graph (one job per GPU; when a GPU frees up,
               the next job starts). Selection on validation AUC only.
  seeds      : retrain each study's best config over --seeds seeds, so the test
               AUCs come with error bars.
  report     : aggregate everything into results/exp10/<dataset>/summary.{md,tex}
               and print it.

Usage (after connecting, e.g.
  srun -p killable --gres=gpu:a5000:2 -c 50 --mem=16G -t 24:00:00 --pty bash
and activating the venv):
  python run_exp10.py                          # all four stages, rgat backbone
  python run_exp10.py --backbones rgat rgcn    # both relational backbones
  python run_exp10.py --stage report           # just rebuild the tables
  python run_exp10.py --trials 5 --seeds 2     # short end-to-end rehearsal

The hybrid study has four extra hyperparameters on top of train09's backbone
space (threshold quantile, k_core, similarity space, and whether the core and
threshold edges form separate relations), so it gets --hybrid-trials (default
150) rather than --trials. To sweep the threshold alone on the exp08
neighbourhoods, pass --sim-spaces raw and drop it back to ~80.

Progress: tail -f logs/<job name>.log
Interrupted study jobs are cheap to rerun: already-tried hp configs are skipped
via results/exp10/<dataset>/<backbone>_<graph>/hp_results.json.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from os.path import join

import pandas as pd

GRAPHS = ["knn20", "hybrid"]

# Row descriptions for the summary table, in the style of the exp08/exp09 reports.
GRAPH_DESC = {
    "knn20": "Control (exp08). Cosine $k$NN at a fixed $k=20$; every node gets exactly 20 neighbours",
    "hybrid": "2.2 $k_{core}$NN core plus every candidate above a similarity threshold $\\tau$; "
              "degree adapts to local density, and the similarity space is swept",
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


def build_candidates(args):
    """
    Serial prerequisite: cache the candidate neighbours, verify them against the
    exp08 edges.txt, and print the density grid the sweep will explore.
    """
    from train10 import K_MAX
    cmd = [sys.executable, "graph_construction.py", "--dataset", args.dataset,
           "--k-max", str(K_MAX), "--verify", "--describe", "--spaces"] + args.sim_spaces
    if args.dataset == "diseases":
        cmd += ["--names"] + args.diseases
    print(" ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def trials_for(args, graph):
    return args.hybrid_trials if graph == "hybrid" else args.trials


def study_jobs(args):
    return [(f"exp10_{b}_{g}",
             [sys.executable, "train10.py", "--graph", g, "--backbone", b,
              "--dataset", args.dataset, "--trials", str(trials_for(args, g)),
              "--epochs", str(args.epochs), "--sim-spaces"] + args.sim_spaces)
            for b in args.backbones for g in args.graphs]


def seed_jobs(args):
    return [(f"exp10_{b}_{g}_seeds",
             [sys.executable, "train10.py", "--graph", g, "--backbone", b,
              "--dataset", args.dataset, "--seed-eval", "--seeds", str(args.seeds),
              "--epochs", str(args.epochs)])
            for b in args.backbones for g in args.graphs]


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


def _graph_summary(hps, row):
    """One-line description of the graph the best config actually used."""
    if hps.get("graph") == "knn20":
        return f"raw, k={hps.get('k', 20)}"
    parts = [hps.get("sim_space", "raw"), f"q={hps['sim_quantile']}", f"k_core={hps['k_core']}"]
    if not pd.isna(row.get("tau")):
        parts.append(f"tau={row['tau']:.4f}")
    parts.append("split" if hps.get("split_core_relation") else "merged")
    return ", ".join(parts)


def collect(dataset, backbones, graphs):
    """
    Build one summary row per (backbone, graph): the best configuration by
    validation AUC, and its test AUC over seeds.
    """
    rows = []
    for backbone in backbones:
        for graph in graphs:
            run_dir = join("results", "exp10", dataset, f"{backbone}_{graph}")
            trial_path = join(run_dir, "trial_results.csv")
            if not os.path.exists(trial_path):
                print(f"  (skipping {backbone}/{graph}: no study results yet)")
                continue

            trials = pd.read_csv(trial_path)
            trials["val_score"] = (trials["val_gnn_auc"] + trials["val_color_auc"]) / 2
            per_config = _mean_over_datasets(trials, ["hps"])
            best = per_config.loc[per_config["val_score"].idxmax()]
            best_hps = json.loads(best["hps"])

            row = {
                "backbone": backbone,
                "graph": graph,
                "description": GRAPH_DESC.get(graph, ""),
                "graph_config": _graph_summary(best_hps, best),
                "mean_degree": best.get("mean_degree", float("nan")),
                "n_edges": best.get("n_edges", float("nan")),
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
            seed_path = join("results", "exp10", dataset, "seed_results.csv")
            if os.path.exists(seed_path):
                seeds = pd.read_csv(seed_path)
                seeds = seeds[(seeds["backbone"] == backbone) & (seeds["graph"] == graph)]
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


def _hybrid_trials(dataset, backbones):
    frames = []
    for backbone in backbones:
        path = join("results", "exp10", dataset, f"{backbone}_hybrid", "trial_results.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df["backbone"] = backbone
        for field in ("sim_quantile", "k_core", "sim_space", "split_core_relation"):
            df[field] = df["hps"].apply(lambda h, f=field: json.loads(h).get(f))
        df["val_score"] = (df["val_gnn_auc"] + df["val_color_auc"]) / 2
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _marginal(df, by):
    """
    Mean and best validation score over every backbone configuration sharing a
    value of `by`. Says whether an axis matters at all, independently of which
    single config happened to win.
    """
    return df.groupby(["backbone"] + by).agg(
        configs=("hps", "nunique"),
        tau=("tau", "mean"),
        mean_degree=("mean_degree", "mean"),
        val_score_mean=("val_score", "mean"),
        val_score_best=("val_score", "max"),
        test_color_best=("test_color_auc", "max"),
    ).reset_index()


def threshold_curve(dataset, backbones):
    """
    The two graph axes, marginalised. The threshold curve is broken down *within*
    each similarity space: a raw tau of 0.999 and a whitened tau of 0.48 are the
    same quantile but not remotely the same number, so averaging across spaces
    would be meaningless.
    """
    df = _hybrid_trials(dataset, backbones)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    return _marginal(df, ["sim_space", "sim_quantile"]), _marginal(df, ["sim_space"])


def _pm(mean, std, seeds):
    return f"{mean:.3f} ± {std:.3f}" if seeds > 1 else f"{mean:.3f}"


def to_markdown(df, curve, space_curve, dataset, seeds_note):
    lines = [f"# Experiment 10 - graph construction ({dataset})", "",
             "Readout frozen at exp08/exp09 mean-pooling, same-color relation untouched, backbone",
             "hyperparameter space identical to exp09. Only the kNN relation varies. Configurations",
             f"selected on validation AUC only; test AUC is never optimized on. {seeds_note}", "",
             "| Backbone | Graph | What was done | Best graph | Mean deg | Configs | "
             "Val AUC (node/color) | Test AUC node | Test AUC color |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['backbone']} | `{r['graph']}` | {r['description']} | {r['graph_config']} | "
            f"{r['mean_degree']:.1f} | {r['n_configs']} | "
            f"{r['val_node']:.3f} / {r['val_color']:.3f} | "
            f"{_pm(r['test_node'], r['test_node_std'], r['n_seeds'])} | "
            f"{_pm(r['test_color'], r['test_color_std'], r['n_seeds'])} |")

    if len(curve):
        lines += ["", "## Threshold sweep", "",
                  "Mean and best validation score over every backbone configuration that used each",
                  "quantile - does graph density matter, independently of which config won? Broken",
                  "down within each space, since tau is not comparable across spaces.", "",
                  "| Backbone | Space | q | tau | Mean deg | Configs | Val (mean) | Val (best) | Test color (best) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for _, r in curve.iterrows():
            lines.append(
                f"| {r['backbone']} | `{r['sim_space']}` | {r['sim_quantile']:.2f} | {r['tau']:.4f} | "
                f"{r['mean_degree']:.1f} | {int(r['configs'])} | {r['val_score_mean']:.3f} | "
                f"{r['val_score_best']:.3f} | {r['test_color_best']:.3f} |")

    if len(space_curve):
        lines += ["", "## Similarity space", "",
                  "The MiniLM embeddings are anisotropic (mean cosine between two random pregnancy",
                  "nodes: +0.979), so `raw` neighbourhoods are decided in the fourth decimal place.",
                  "`centered` and `whitened` strip that common component.", "",
                  "| Backbone | Space | Mean deg | Configs | Val (mean) | Val (best) | Test color (best) |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in space_curve.iterrows():
            lines.append(
                f"| {r['backbone']} | `{r['sim_space']}` | {r['mean_degree']:.1f} | "
                f"{int(r['configs'])} | {r['val_score_mean']:.3f} | "
                f"{r['val_score_best']:.3f} | {r['test_color_best']:.3f} |")

    lines += ["", "## Best configuration per graph", ""]
    for _, r in df.iterrows():
        lines.append(f"- **{r['backbone']}/{r['graph']}**: `{r['best_hps']}`")
    return "\n".join(lines) + "\n"


def to_latex(df, dataset, seeds_note):
    """A booktabs table in the format of the exp08/exp09 reports, ready to \\input."""
    esc = lambda s: str(s).replace("_", r"\_")
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\begin{tabularx}{\textwidth}{llXlcc}", r"\toprule",
             r"Backbone & Graph & What was done & \makecell{Best graph\\(mean deg)} "
             r"& \makecell{Val AUC\\(node/color)} & \makecell{Test AUC\\node\\color} \\", r"\midrule"]
    for _, r in df.iterrows():
        node = _pm(r["test_node"], r["test_node_std"], r["n_seeds"]).replace("±", r"$\pm$")
        color = _pm(r["test_color"], r["test_color_std"], r["n_seeds"]).replace("±", r"$\pm$")
        config = f"{esc(r['graph_config'])} ({r['mean_degree']:.1f})"
        lines.append(
            f"{esc(r['backbone'])} & \\texttt{{{esc(r['graph'])}}} & {r['description']} & "
            f"{config} & {r['val_node']:.3f} / {r['val_color']:.3f} & "
            f"\\makecell{{{node}\\\\{color}}} \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}",
              rf"\caption{{\texttt{{{esc(dataset)}}}. Best configuration per graph, selected by "
              rf"validation AUC. {esc(seeds_note)}}}",
              r"\label{tab:exp10-graph}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def report(dataset, backbones, graphs):
    print(f"\nBuilding the summary for {dataset}...")
    df = collect(dataset, backbones, graphs)
    if df.empty:
        print("No results found under results/exp10 - run the studies first.")
        return

    curve, space_curve = threshold_curve(dataset, backbones)
    n_seeds = int(df["n_seeds"].max())
    seeds_note = (f"Test AUC is the mean ± standard deviation over {n_seeds} seeds "
                  f"(only model init and dropout vary; the split is fixed)."
                  if n_seeds > 1 else
                  "Test AUC is from the single selection run - run the seed stage for error bars.")

    out_dir = join("results", "exp10", dataset)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(join(out_dir, "summary.csv"), index=False)
    if len(curve):
        curve.to_csv(join(out_dir, "threshold_curve.csv"), index=False)
        space_curve.to_csv(join(out_dir, "space_curve.csv"), index=False)
    md = to_markdown(df, curve, space_curve, dataset, seeds_note)
    with open(join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(join(out_dir, "summary.tex"), "w", encoding="utf-8") as f:
        f.write(to_latex(df, dataset, seeds_note))

    print("\n" + md)

    control = df[df["graph"] == "knn20"]
    if len(control):
        base_node, base_color = control.iloc[0]["test_node"], control.iloc[0]["test_color"]
        print("Change vs. the k=20 control, in AUC points:")
        for _, r in df.iterrows():
            if r["graph"] == "knn20":
                continue
            print(f"  {r['backbone']}/{r['graph']:<10s} "
                  f"node {100 * (r['test_node'] - base_node):+.1f}   "
                  f"color {100 * (r['test_color'] - base_color):+.1f}")

    print(f"\nWritten: {join(out_dir, 'summary.md')}, summary.tex, summary.csv")


def main():
    parser = argparse.ArgumentParser(description="Run and report the exp10 graph-construction comparison.")
    parser.add_argument("--dataset", type=str, default="pregnancy", choices=["pregnancy", "diseases"])
    parser.add_argument("--backbones", type=str, nargs="+", default=["rgat"], choices=["rgat", "rgcn"])
    parser.add_argument("--graphs", type=str, nargs="+", default=GRAPHS, choices=GRAPHS)
    parser.add_argument("--trials", type=int, default=60, help="Optuna trials for the knn20 control")
    parser.add_argument("--hybrid-trials", type=int, default=150,
                        help="Optuna trials for the hybrid graph, which has four extra hyperparameters")
    parser.add_argument("--sim-spaces", type=str, nargs="+",
                        default=["raw", "centered", "whitened"],
                        choices=["raw", "centered", "whitened"],
                        help="Pass just 'raw' to sweep the threshold alone, on the exp08 neighbourhoods.")
    parser.add_argument("--seeds", type=int, default=5, help="Seeds per best configuration")
    parser.add_argument("--epochs", type=int, default=10000,
                        help="Epoch cap passed to train10.py. The default relies on early stopping.")
    parser.add_argument("--diseases", type=str, nargs="+", default=None)
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "candidates", "studies", "seeds", "report"])
    args = parser.parse_args()

    if args.diseases is None:
        from train08 import DISEASES
        args.diseases = DISEASES

    failed = []
    if args.stage in ("all", "candidates"):
        print("=== Stage 1/4: candidate neighbours ===")
        if build_candidates(args) != 0:
            print("Candidate build failed - stopping.")
            return

    if args.stage in ("all", "studies"):
        print("\n=== Stage 2/4: hyperparameter studies ===")
        failed += run_jobs(study_jobs(args))

    if args.stage in ("all", "seeds"):
        if failed:
            print(f"Studies failed: {failed} - fix them before the seed stage, "
                  f"otherwise the best configs are selected from partial results.")
            return
        print("\n=== Stage 3/4: multi-seed evaluation ===")
        failed += run_jobs(seed_jobs(args))

    if args.stage in ("all", "report"):
        print("\n=== Stage 4/4: report ===")
        report(args.dataset, args.backbones, args.graphs)

    if failed:
        print(f"\nFailed jobs: {failed} - check logs/<name>.log and rerun.")
    else:
        print("\nAll done. Export the results/exp10/ folder.")


if __name__ == "__main__":
    main()
