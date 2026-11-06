"""
Yuli Tshuva.
run_edge_colors.py - which edge colors, aggregation, pooling and training make
the copies graph work, on GPEN's eight benchmarks.

The graph: every shared node (in two or more subgraphs) is replaced by one copy
per subgraph, each with all of the node's graph edges. Its edges come in kinds
(run_gpen_compare.copy_edge_kinds):
  Icopy, Iplain : inside edges (both ends in one subgraph) that touch a copy / that do not
  Osub, Onone   : outside edges between members of two subgraphs / with an end in no subgraph
  Oin, Oout     : the outside edges' messages, by whether the receiver is in a subgraph
  S             : sibling edges, between two copies of one node
  C             : every pair of members of one subgraph (our clique)
  ("I" is Icopy + Iplain, "O" is Osub + Onone)
A coloring groups kinds into colors; each color is one relation of the RGNN, with
its own weights. A kind in no color is deleted from the graph. COLORINGS holds the
14 base colorings, each one's version without sibling edges when that is a new
graph, and two colorings of the graph without copies (the baselines; kinds Gin,
Gout = edge_label's inside and outside edges, "G" both): 23 in all.

Steps, each one finished for a dataset before its next one starts:
  1. every coloring x both trainings (ours, GLASS's recipe), with GLASS's aggregation
     and pooling for the dataset, 3 seeds each. em_user, hpo_metab and hpo_neuro
     skip every coloring with copies (NO_COPIES): their copies graphs have 35M-1.3B
     edges and run out of GPU memory, so they run the two baselines only.
  2. 4 aggregations (mean, sum, gcn, attention = RGAT) x 7 poolings (multistat, mean,
     max, sum, size = sum / sqrt(size), gated attention, Set Transformer) on step 1's
     3 best colorings, under step 1's better training, 3 seeds each.
  3. tuning: the 3 best setups of steps 1-2, each on a grid (TUNE_GRID): hidden size
     (GLASS's or 128), lr (x1/3, x1, x3), dropout (0 or 0.5) and, on the synthetic
     datasets, the edge-count features (off or on), 3 seeds each.
  4. each of the 3 setups at its best grid point (or untuned, if that was better),
     plus the reference, our previous best config for the dataset (REFERENCE), on
     10 seeds. final.csv reports the best of the three by validation next to it.
Selection is on validation micro-F1, the mean over the first 3 seeds; test is only
reported. Every run is on GLASS's splits (glass_split), seed k on GLASS's repeat k.

Model, the same in every run before tuning: GLASS's architecture for the dataset
(its hidden size, number of layers, jumping knowledge); input GLASS's pretrained
vectors, fine-tuned, on the real datasets and the structural features on the
synthetic ones; no edge-count features (step 3 tries them).
Training:
  ours : our best training so far for the dataset (edge_label on the real datasets,
         copies_skip_tune on the synthetic ones, pool where neither ran): its lr,
         batches, norm, dropout, MLP head and node head; early stopping on validation
         micro-F1 and lr decay on the validation loss.
  glass: GLASS's recipe (run_gpen_compare.glass_recipe): its batch size, lr, decay on
         the training loss, GraphNorm, input and message dropout, a linear head, its
         early stopping, no node head.

Resuming: every finished seed is a row in its config's CSV, and every step's configs
are worked out again from those rows, so after a stop (time limit, preemption, crash)
the same command picks up where it was: finished seeds are kept, a seed cut off
mid-training is redone.

Usage (server):
  python run_edge_colors.py --max-parallel 4         # steps 1-4, every dataset, over the GPUs
  python run_edge_colors.py --steps 1 --max-parallel 4
  python run_edge_colors.py --exclude em_user
  python run_edge_colors.py --dataset density --smoke   # every step, 1-2 seeds, 15 epochs
  python run_edge_colors.py --collect                # rebuild the summaries only

Results (results/edge_colors/):
  runs/<dataset>/<config>.csv   <- one row per seed (steps share a config's file;
                                   step 4 adds seeds 3-9 to it)
  logs/<dataset>/<config>.log
  summary.csv                   <- one row per (dataset, config), on the first 3 seeds
                                   (and on all seeds when step 4 added more)
  sibling_effect.csv            <- each step-1 coloring with and without its sibling edges
  final.csv                     <- per dataset the best step-4 setup and the reference,
                                   test mean +- std over 10 seeds
"""

# Imports
from run_gpen_compare import (DATASETS, REAL, HEAVIEST_FIRST, GLASS_CONFIGS, KIND_GROUPS, GRAPH_KINDS,
                              load_dataset, prepare, glass_split, glass_recipe, build_colored_relations,
                              build_copy_edges, build_member_pairs, train, case_dir, make_row, append_row,
                              is_oom, free_memory)
from train08 import EPOCHS
from os.path import join
import argparse
import glob
import json
import os
import pandas as pd
import subprocess
import sys
import time
import torch

# Colors are separated by "_" in a coloring's name, kinds within a color by "+"
BASE_COLORINGS = [
    [["I", "O", "S"]],  # one color
    [["S"], ["I", "O"]],
    [["I"], ["O", "S"]],
    [["O"], ["I", "S"]],
    [["I"], ["O"], ["S"]],  # build_copy_relations' three relations
    [["I"], ["O"]],
    [["I"], ["Osub"], ["Onone"], ["S"]],
    [["I", "Osub"], ["Onone"], ["S"]],  # anything touching another subgraph vs truly outside
    [["Icopy"], ["Iplain"], ["O"], ["S"]],
    [["I"], ["Oin"], ["Oout"], ["S"]],  # outside messages split by direction
    [["Icopy"], ["Iplain"], ["Osub"], ["Onone"], ["S"]],  # everything apart
    [["I", "O"]],  # the plain graph after copying
    [["I"], ["O"], ["S"], ["C"]],  # with our clique
    [["I", "O", "S"], ["C"]],
]
# The baselines without copies: the dataset's graph as is, and split as edge_label splits it
NO_COPY_COLORINGS = [[["G"]], [["Gin"], ["Gout"]]]


def coloring_name(coloring):
    return "_".join("+".join(color) for color in coloring)


def canonical(coloring):
    """Two colorings with the same canonical form build the same graph."""
    return frozenset(frozenset(k for name in color for k in KIND_GROUPS.get(name, (name,))) for color in coloring)


def without_siblings(coloring):
    return [kept for color in coloring if (kept := [k for k in color if k != "S"])]


def uses_copies(coloring):
    return not any(k in GRAPH_KINDS for color in coloring for name in color for k in KIND_GROUPS.get(name, (name,)))


def all_colorings():
    """The base colorings, each one's version without sibling edges that is a new graph, the baselines."""
    colorings, seen = list(BASE_COLORINGS), {canonical(c) for c in BASE_COLORINGS}
    for c in BASE_COLORINGS:
        bare = without_siblings(c)
        if canonical(bare) not in seen:
            colorings.append(bare)
            seen.add(canonical(bare))
    return {coloring_name(c): c for c in colorings + NO_COPY_COLORINGS}


COLORINGS = all_colorings()
TRAININGS = ["ours", "glass"]
# Copies graph too big for a 24 GB GPU (hpo: 464M / 1.3B edges, em_user 35M)
NO_COPIES = ("em_user", "hpo_metab", "hpo_neuro")
AGGREGATIONS = ["mean", "sum", "gcn", "attention"]
POOLINGS = ["multistat", "mean", "max", "sum", "size", "gated_attn", "transformer"]
# Step 3's grid. hidden "glass": GLASS's size for the dataset; lr: a factor on the training's own lr;
# counts: copies_plus' per-copy inside/outside counts as input and copies_skip's per-subgraph counts
# at the head (only the latter without copies), synthetic datasets only
TUNE_GRID = {"hidden": ["glass", "128"], "lr": ["0.33", "1", "3"], "dropout": ["0", "0.5"], "counts": ["0", "1"]}
STEP_SEEDS, SMOKE_SEEDS = {1: 3, 2: 3, 3: 3, 4: 10}, {1: 1, 2: 1, 3: 1, 4: 2}
RANK_SEEDS = 3  # configs are ranked on their first seeds only, so every config gets the same number
TOP_COLORINGS, TOP_SETUPS = 3, 3
# The case of run_gpen_compare whose best config gives "ours" for the dataset; pool when it has none
OURS_SOURCE = {"real": "edge_label", "synthetic": "copies_skip_tune"}
OURS_KEYS = {"lr": 1e-3, "n_batches": 1, "norm": None, "gnn_dropout_rate": 0.0, "mlp_dropout_rate": 0.0,
             "weight_decay": 0.0, "color_mlp_hidden_dims": 64, "node_loss_weight": 1.0}  # with defaults
# Our previous best test micro-F1 per dataset and its case, rerun in step 4 as the reference
REFERENCE = {"ppi_bp": "edge_label_glass",  # 64.0
             "hpo_metab": "edge_label",  # 59.7
             "hpo_neuro": "glass_plain",  # 68.7
             "em_user": "glass_plain",  # 80.8
             "density": "copies_recipe",  # 95.4
             "cut_ratio": "copies_recipe",  # 94.1
             "coreness": "transformer",  # 89.1
             "component": "copies_skip"}  # 100.0
RESULTS = join("results", "edge_colors")


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------
# A key is <coloring>__<training>__<aggregation>__<pooling>, plus __<grid point> once tuned
# (h<hidden>_lr<factor>_d<dropout>_c<counts>), or reference__<case>.

def config_key(coloring, training, agg, pool):
    return f"{coloring}__{training}__{agg}__{pool}"


def tune_points(name):
    counts = TUNE_GRID["counts"] if name not in REAL else ["0"]
    return [f"h{h}_lr{lr}_d{d}_c{c}" for h in TUNE_GRID["hidden"] for lr in TUNE_GRID["lr"]
            for d in TUNE_GRID["dropout"] for c in counts]


def tuned_keys(name, key):
    return [f"{key}__{point}" for point in tune_points(name)]


def parse_key(key):
    """:return: dict of the key's parts (coloring, training, aggregation, pooling, tune or reference)"""
    parts = key.split("__")
    if parts[0] == "reference":
        return {"reference": parts[1], "coloring": None, "training": None, "aggregation": None,
                "pooling": None, "tune": None}
    return {"reference": None, "coloring": parts[0], "training": parts[1], "aggregation": parts[2],
            "pooling": parts[3], "tune": parts[4] if len(parts) == 5 else None}


def parse_tune(point):
    h, lr, d, c = point.split("_")
    return {"hidden": h[1:], "lr": float(lr[2:]), "dropout": float(d[1:]), "counts": c[1:] == "1"}


def glass_agg_pool(name):
    """GLASS's aggregation and pooling for the dataset (its config file)."""
    return GLASS_CONFIGS[name]["aggr"], GLASS_CONFIGS[name]["pool"]


def training_hps(name, training):
    if training == "glass":
        recipe = {k: v for k, v in glass_recipe(name).items() if k != "readout"}
        return {**recipe, "gnn_dropout_rate": GLASS_CONFIGS[name]["dropout"], "mlp_dropout_rate": 0.0,
                "weight_decay": 0.0, "node_loss_weight": 0.0}
    source = OURS_SOURCE["real" if name in REAL else "synthetic"]
    base = {}
    for case in (source, "pool"):
        path = join(case_dir(case), name, "summary.json")
        if os.path.exists(path):
            with open(path) as f:
                base = json.loads(json.load(f)["hps"])
            break
    return {**{k: base.get(k, default) for k, default in OURS_KEYS.items()},
            "batch_size": None, "lr_on": "val", "glass_stop": False, "input_norm": None, "input_dropout": 0.0,
            "msg_dropout": 0.0}


def reference_hps(name):
    """Our previous best config for the dataset, None if its result is missing."""
    path = join(case_dir(REFERENCE[name]), name, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        hps = json.loads(json.load(f)["hps"])
    if hps.get("copies"):
        hps["copy_all"] = True  # its copies were the old ones, a copy of every member
    return hps


def config_hps(name, key):
    p = parse_key(key)
    if p["reference"]:
        return reference_hps(name)
    coloring = COLORINGS[p["coloring"]]
    copies = uses_copies(coloring)
    hidden, layers = GLASS_CONFIGS[name]["hidden_dim"], GLASS_CONFIGS[name]["conv_layer"]
    tune = parse_tune(p["tune"]) if p["tune"] else None
    if tune and tune["hidden"] != "glass":
        hidden = int(tune["hidden"])
    # The Set Transformer's heads must divide the pooled width (jumping knowledge: hidden x layers);
    # RGAT's 4 heads act only on hidden layers, whose widths (20, 64, 128) they divide
    pma_heads = next(h for h in (4, 2, 1) if hidden * layers % h == 0)
    hps = {"coloring": coloring, "copies": copies, "copy_clique": False,
           "backbone": "rgat" if p["aggregation"] == "attention" else "rgcn", "num_heads": 4,
           "graph_agg": "mean" if p["aggregation"] == "attention" else p["aggregation"], "readout": p["pooling"],
           "gnn_embedding_dim": hidden, "gnn_hidden_dim": hidden, "k_gnn_layers": layers, "jk": True,
           "gnn_mlp_hidden_dims": 64, "node_input": "glass_ft" if name in REAL else "struct",
           "feature_norm": "none", "attn_pool_hidden_dim": 32, "attn_pool_dropout": 0.0,
           "pma_heads": pma_heads, "pma_sabs": 1, **training_hps(name, p["training"])}
    if tune:
        hps["lr"] *= tune["lr"]
        hps["gnn_dropout_rate"] = tune["dropout"]
        if p["training"] == "glass":
            # GLASS's recipe drops out the input and the messages by the same rate
            hps["input_dropout"] = hps["msg_dropout"] = tune["dropout"]
        if tune["counts"]:
            hps["count_skip"] = True
            hps["copy_counts"] = copies
    return hps


def colorings_for(name):
    return [c for c, spec in COLORINGS.items() if not (name in NO_COPIES and uses_copies(spec))]


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def runs_path(root, name, key):
    return join(root, "runs", name, f"{key}.csv")


def read_runs(root, name, key):
    """
    A config's seeds, or None. A row cut off by a kill mid-write is dropped, so
    its seed counts as not done and is rerun.
    """
    path = runs_path(root, name, key)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, on_bad_lines="skip").dropna(subset=["seed", "val_f1", "test_f1", "hps"])
    df["seed"] = df["seed"].astype(int)
    return df.drop_duplicates("seed", keep="last")


def seeds_done(root, name, key):
    df = read_runs(root, name, key)
    return set() if df is None else set(df["seed"])


def is_done(root, name, key, n_seeds):
    return set(range(n_seeds)) <= seeds_done(root, name, key)


def mean_val(root, name, key):
    """Mean validation micro-F1 over the first RANK_SEEDS seeds; None for an OOM config."""
    df = read_runs(root, name, key)
    df = df[df["seed"] < RANK_SEEDS]
    return None if (df["val_f1"] < 0).any() else float(df["val_f1"].mean())


def ranked(root, name, keys):
    """keys by mean validation micro-F1, best first (ties by name); OOM configs left out."""
    scores = {k: mean_val(root, name, k) for k in keys}
    return sorted((k for k in keys if scores[k] is not None), key=lambda k: (-scores[k], k))


def step_configs(step, name, root, seeds):
    """
    The config keys of a step for a dataset, or None while the previous step is
    unfinished (it decides this step's configs). Worked out from the result files
    alone, so a stopped run resumes with the same configs.
    """
    agg, pool = glass_agg_pool(name)
    step1 = [config_key(c, t, agg, pool) for c in colorings_for(name) for t in TRAININGS]
    if step == 1:
        return step1
    if not all(is_done(root, name, k, seeds[1]) for k in step1):
        return None
    best1 = ranked(root, name, step1)
    if not best1:
        return []
    training = parse_key(best1[0])["training"]
    top = [parse_key(k)["coloring"] for k in best1 if parse_key(k)["training"] == training][:TOP_COLORINGS]
    step2 = [config_key(c, training, a, p) for c in top for a in AGGREGATIONS for p in POOLINGS]
    if step == 2:
        return step2
    if not all(is_done(root, name, k, seeds[2]) for k in step2):
        return None
    finalists = ranked(root, name, list(dict.fromkeys(step1 + step2)))[:TOP_SETUPS]
    step3 = [k for f in finalists for k in tuned_keys(name, f)]
    if step == 3:
        return step3
    if not all(is_done(root, name, k, seeds[3]) for k in step3):
        return None
    # Each finalist at its best grid point, or untuned when no point beat it
    best = [r[0] for f in finalists if (r := ranked(root, name, [f] + tuned_keys(name, f)))]
    reference = [f"reference__{REFERENCE[name]}"] if reference_hps(name) is not None else []
    return list(dict.fromkeys(best)) + reference


# ---------------------------------------------------------------------------
# One config
# ---------------------------------------------------------------------------

def run_config(name, key, n_seeds, root, data_root, epochs):
    """Train one config on seeds 0..n_seeds-1, skipping those already in its CSV."""
    path = runs_path(root, name, key)
    todo = sorted(set(range(n_seeds)) - seeds_done(root, name, key))
    if not todo:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        # A row cut off mid-write has no newline; end it, so the next row starts on a line of its own
        with open(path, "rb+") as f:
            f.seek(0, os.SEEK_END)
            if f.tell() > 0:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    f.write(b"\n")
    start = time.time()
    prep = prepare(load_dataset(name, data_root))
    prep["name"] = name
    hps = config_hps(name, key)
    # Build the cached parts on the base prep, so the per-seed split copies share them
    if hps.get("copies") or hps.get("count_skip"):
        prep["copy_edges"] = build_copy_edges(prep["subgraphs"], prep["graph_edges"], prep["n_nodes"])
    if hps.get("coloring"):
        build_colored_relations(prep, hps)
    if hps["readout"] in ("pma", "transformer") and hps.get("pma_sabs", 0) > 0:
        prep["member_pairs"] = build_member_pairs(prep["member_sub"])
    print(f"[{name}] {key}: loaded in {time.time() - start:.1f}s, seeds {todo}\n{hps}")
    for i, seed in enumerate(todo):
        start = time.time()
        try:
            res = train(glass_split(prep, name, seed), hps, epochs=epochs, seed=seed)
        except RuntimeError as e:
            if not is_oom(e):
                raise
            # Recorded with -1 on every remaining seed, so the config is never retried and ranks last
            print(f"[{name}] {key}: OOM - recording seeds {todo[i:]} as -1")
            for s in todo[i:]:
                append_row(path, make_row(
                    name, hps, key, s, {"train_f1": -1.0, "val_f1": -1.0, "test_f1": -1.0, "epochs": 0}, 0.0))
            return
        append_row(path, make_row(name, hps, key, seed, res, time.time() - start))
        free_memory()
        print(f"[{name}] {key} seed {seed}: val {res['val_f1']:.4f}, test {res['test_f1']:.4f} "
              f"({res['epochs']} epochs, {time.time() - start:.0f}s)")


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

def pending_jobs(names, steps, root, seeds):
    """(step, dataset, key, n_seeds) of every config ready to run, heaviest datasets first."""
    jobs = []
    for name in names:
        for step in steps:
            keys = step_configs(step, name, root, seeds)
            if keys is None:
                break
            todo = [k for k in keys if not is_done(root, name, k, seeds[step])]
            jobs += [(step, name, k, seeds[step]) for k in todo]
            if todo:
                break  # the next step waits for this one
    return jobs


def run_all(args, root, seeds, epochs):
    """
    Every ready config as its own child process, one per GPU slot, as in
    run_gpen_compare.run_all. A step's configs for a dataset are queued once its
    previous step is done, so later steps of one dataset run beside step 1 of another.
    """
    names = [n for n in HEAVIEST_FIRST if n not in args.exclude] if args.dataset == "all" else [args.dataset]
    n_gpus = torch.cuda.device_count()
    slots = [str(g) for _ in range(args.jobs_per_gpu) for g in range(n_gpus)][:args.max_parallel] or [None]
    running, failed = {}, set()
    print(f"{len(slots)} at a time on {n_gpus} GPU(s); steps {args.steps}; datasets {names}")
    while True:
        busy = {i for i, *_ in running.values()}
        active = {job for *_, job in running.values()}
        queue = [j for j in pending_jobs(names, args.steps, root, seeds) if j[1:3] not in active | failed]
        if not queue and not running:
            break
        for i in [i for i in range(len(slots)) if i not in busy]:
            if not queue:
                break
            step, name, key, n_seeds = queue.pop(0)
            env = dict(os.environ)
            if slots[i] is not None:
                env["CUDA_VISIBLE_DEVICES"] = slots[i]
            os.makedirs(join(root, "logs", name), exist_ok=True)
            log = open(join(root, "logs", name, f"{key}.log"), "a")
            cmd = [sys.executable, os.path.abspath(__file__), "--dataset", name, "--run-config", key,
                   "--n-seeds", str(n_seeds), "--data-root", args.data_root, "--epochs", str(epochs)]
            proc = subprocess.Popen(cmd + (["--smoke"] if args.smoke else []), env=env, stdout=log,
                                    stderr=subprocess.STDOUT)
            running[proc.pid] = (i, proc, log, (name, key))
            print(f"  step {step} started {name}/{key} on GPU {slots[i]}")
        time.sleep(5)
        for pid, (i, proc, log, job) in list(running.items()):
            if proc.poll() is None:
                continue
            log.close()
            if proc.returncode != 0:
                failed.add(job)
                print(f"  FAILED {job[0]}/{job[1]} (exit code {proc.returncode})")
            del running[pid]
    collect(root, seeds)
    if failed:
        print(f"FAILED (their datasets' next steps never started): {sorted(failed)} - see logs/")


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def collect(root, seeds):
    """summary.csv, sibling_effect.csv and final.csv from every config's runs."""
    rows = []
    for path in sorted(glob.glob(join(root, "runs", "*", "*.csv"))):
        name, key = os.path.basename(os.path.dirname(path)), os.path.basename(path)[:-4]
        df = read_runs(root, name, key)
        if df.empty:
            continue
        # A row on the ranking seeds, and one on all seeds when step 4 added more
        parts = [(f"<{RANK_SEEDS}", df[df["seed"] < RANK_SEEDS])]
        if len(df) > len(parts[0][1]):
            parts.append(("all", df))
        for seeds_label, part in parts:
            rows.append({"dataset": name, "config": key, **parse_key(key), "seeds": seeds_label,
                         "n_seeds": len(part), "oom": bool((part["val_f1"] < 0).any()),
                         "val_mean": round(part["val_f1"].mean() * 100, 2),
                         "val_std": round(part["val_f1"].std(ddof=0) * 100, 2),
                         "test_mean": round(part["test_f1"].mean() * 100, 2),
                         "test_std": round(part["test_f1"].std(ddof=0) * 100, 2),
                         "train_mean": round(part["train_f1"].mean() * 100, 2),
                         "epochs_mean": round(part["epochs"].mean(), 1),
                         "seconds_mean": round(part["seconds"].mean(), 1)})
    if not rows:
        print(f"No runs under {root}")
        return
    summary = pd.DataFrame(rows)
    summary.to_csv(join(root, "summary.csv"), index=False)

    # Sibling edges: each step-1 coloring with S against the same coloring without it
    first = summary[summary["seeds"] == f"<{RANK_SEEDS}"].set_index(["dataset", "config"])
    by_form = {canonical(spec): c for c, spec in COLORINGS.items()}
    effect = []
    for (name, key), row in first.iterrows():
        p = parse_key(key)
        if p["reference"] or p["tune"] or (p["aggregation"], p["pooling"]) != glass_agg_pool(name) \
                or not any("S" in col for col in COLORINGS[p["coloring"]]):
            continue
        bare = config_key(by_form[canonical(without_siblings(COLORINGS[p["coloring"]]))], p["training"],
                          p["aggregation"], p["pooling"])
        if (name, bare) in first.index:
            other = first.loc[(name, bare)]
            effect.append({"dataset": name, "training": p["training"], "with_siblings": p["coloring"],
                           "without_siblings": parse_key(bare)["coloring"],
                           "val_with": row["val_mean"], "val_without": other["val_mean"],
                           "test_with": row["test_mean"], "test_without": other["test_mean"]})
    if effect:
        pd.DataFrame(effect).to_csv(join(root, "sibling_effect.csv"), index=False)

    # Final: per dataset, the step-4 setup with the best validation over all its seeds, and the reference
    final = []
    for name in DATASETS:
        keys = step_configs(4, name, root, seeds)
        if not keys or not all(is_done(root, name, k, seeds[4]) for k in keys):
            continue
        part = summary[(summary["dataset"] == name) & summary["config"].isin(keys)
                       & (summary["n_seeds"] >= seeds[4])]
        ours = part[part["reference"].isna()]
        if not ours.empty:
            final.append({"role": "best", **ours.sort_values("val_mean", ascending=False).iloc[0].to_dict()})
        for _, row in part[part["reference"].notna()].iterrows():
            final.append({"role": "reference", **row.to_dict()})
    if final:
        final = pd.DataFrame(final)
        final.to_csv(join(root, "final.csv"), index=False)
        print(final[["dataset", "role", "config", "val_mean", "test_mean", "test_std", "n_seeds"]]
              .to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Edge colors x aggregation x pooling x training on the copies graph.")
    parser.add_argument("--dataset", default="all", choices=DATASETS + ["all"])
    parser.add_argument("--exclude", nargs="+", default=[], choices=DATASETS)
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 3, 4], choices=[1, 2, 3, 4])
    parser.add_argument("--max-parallel", type=int, default=2,
                        help="Configs running at once; each holds a dataset in RAM (~3 GB)")
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--data-root", default=join("data", "raw"))
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Epoch cap; early stopping decides")
    parser.add_argument("--smoke", action="store_true", help="1-2 seeds, 15 epochs, into results/edge_colors_smoke/")
    parser.add_argument("--collect", action="store_true", help="Only rebuild the summaries")
    # Internal: one config in this process (run_all's children)
    parser.add_argument("--run-config", help=argparse.SUPPRESS)
    parser.add_argument("--n-seeds", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    root = RESULTS + ("_smoke" if args.smoke else "")
    seeds = SMOKE_SEEDS if args.smoke else STEP_SEEDS
    epochs = 15 if args.smoke else args.epochs
    if args.collect:
        collect(root, seeds)
    elif args.run_config:
        run_config(args.dataset, args.run_config, args.n_seeds, root, args.data_root, epochs)
    else:
        run_all(args, root, seeds, epochs)


if __name__ == "__main__":
    main()
