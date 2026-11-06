"""
Yuli Tshuva.
run_edge_colors_copies.py - run_edge_colors' four steps on em_user, hpo_metab and
hpo_neuro WITH the copies colorings, which did not fit before (35M-1.3B outside
edges). The outside edges are computed without building them
(copies_factored.py): the same numbers as the full copies graph, at the memory
of the plain graph. Everything else is run_edge_colors as is.

Differences from run_edge_colors:
  - only these three datasets, results in results/edge_colors_copies/
  - their old runs (the no-copies colorings G, Gin_Gout and the reference) are
    copied from results/edge_colors/ on the first start, so they are not rerun
  - attention (RGAT) is not exact on the factored graph: a copies coloring with
    attention is recorded as -1 (like an OOM), so it ranks last and is skipped

Usage (server, in cc/ with the venv active):
  python run_edge_colors_copies.py --check ppi_bp        # factored == full copies graph, on a small dataset
  python run_edge_colors_copies.py --dataset hpo_metab --smoke --max-parallel 4
  python run_edge_colors_copies.py --max-parallel 4      # all four steps; rerun the same to resume
  python run_edge_colors_copies.py --collect
"""

# Imports
import copies_factored
copies_factored.patch()
import run_edge_colors as rec
import run_gpen_compare as rgc
from os.path import join
import glob
import os
import shutil
import sys
import torch

DATASETS = ["em_user", "hpo_neuro", "hpo_metab"]  # heaviest first
OLD_RESULTS = join("results", "edge_colors")

rec.__file__ = os.path.abspath(__file__)  # run_all's children run this script, with the patches
rec.RESULTS = join("results", "edge_colors_copies")
rec.NO_COPIES = ()
rec.HEAVIEST_FIRST = DATASETS

_build_colored, _run_config = rec.build_colored_relations, rec.run_config


def build_colored_relations(prep, hps):
    """run_config's cache warm-up: the factored relations instead of the full outside edges."""
    if copies_factored.needs_factoring(hps):
        return copies_factored.build_factored_relations(prep, hps)
    return _build_colored(prep, hps)


def run_config(name, key, n_seeds, root, data_root, epochs):
    p = rec.parse_key(key)
    if p["aggregation"] == "attention" and rec.uses_copies(rec.COLORINGS[p["coloring"]]):
        path = rec.runs_path(root, name, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        hps = rec.config_hps(name, key)
        for s in sorted(set(range(n_seeds)) - rec.seeds_done(root, name, key)):
            rgc.append_row(path, rgc.make_row(
                name, hps, key, s, {"train_f1": -1.0, "val_f1": -1.0, "test_f1": -1.0, "epochs": 0}, 0.0))
        print(f"[{name}] {key}: attention on copies is not supported - recorded as -1")
        return
    _run_config(name, key, n_seeds, root, data_root, epochs)


rec.build_colored_relations = build_colored_relations
rec.run_config = run_config


def seed_old_runs(root):
    """Copy the three datasets' finished runs from results/edge_colors/, once."""
    for name in DATASETS:
        dst = join(root, "runs", name)
        if os.path.isdir(dst):
            continue
        os.makedirs(dst)
        for path in glob.glob(join(OLD_RESULTS, "runs", name, "*.csv")):
            shutil.copy(path, dst)
        print(f"{name}: copied {len(os.listdir(dst))} finished configs from {OLD_RESULTS}")


def check(name, data_root):
    """
    Every copies coloring with outside edges x mean/sum/gcn: one random RGCN layer
    on the full copies graph and on the factored one, largest difference printed.
    """
    torch.manual_seed(0)
    prep = rgc.prepare(rgc.load_dataset(name, data_root))
    prep["copy_edges"] = rgc.build_copy_edges(prep["subgraphs"], prep["graph_edges"], prep["n_nodes"])
    dev = rgc.DEVICE
    worst = 0.0
    for cname, coloring in rec.COLORINGS.items():
        if not copies_factored.needs_factoring({"coloring": coloring, "copies": rec.uses_copies(coloring)}):
            continue
        for agg in ("mean", "sum", "gcn"):
            hps = {"coloring": coloring, "copies": True, "graph_agg": agg}
            full = rgc.build_colored_relations(prep, hps)
            fact = copies_factored.build_factored_relations(prep, hps)
            layer = rgc.RGCNLayer(16, 8, full[2], directed=True, sum_relations=full[4]).to(dev)
            n_rows = fact[0].n_rows
            x = torch.randn(n_rows, 16, device=dev)
            w = None if full[3] is None else full[3].to(dev)
            a = layer(x, full[0].to(dev), full[1].to(dev), w)
            b = layer(x, fact[0].to(dev), fact[1].to(dev), None)
            diff = ((a - b).abs().max() / a.abs().max().clamp(min=1e-12)).item()
            worst = max(worst, diff)
            print(f"{cname:32s} {agg:5s} edges {full[0].size(1):>11,d}  relative max diff {diff:.2e}")
    print(f"WORST {worst:.2e} -> {'OK' if worst < 1e-4 else 'MISMATCH'}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        i = sys.argv.index("--check")
        check(sys.argv[i + 1], sys.argv[sys.argv.index("--data-root") + 1] if "--data-root" in sys.argv
              else join("data", "raw"))
        sys.exit()
    if "--run-config" not in sys.argv and "--collect" not in sys.argv:
        seed_old_runs(rec.RESULTS + ("_smoke" if "--smoke" in sys.argv else ""))
    rec.main()
