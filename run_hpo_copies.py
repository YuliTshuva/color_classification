"""
Yuli Tshuva.
run_hpo_copies.py - run_hpo's search on the 4 real datasets with copies. The
coloring is part of the search: I_O (the synthetic datasets' one), Icopy_Iplain_O or
I_O_C (COLORING_CHOICES). The synthetic results stay in results/hpo/.

The copies graphs' outside edges are computed without building them
(copies_factored.py, exact for mean / sum / gcn). Attention is not exact there, so
it is left out of the search space. Everything else is run_hpo as is.

Usage (server, in cc/ with the venv active):
  python run_hpo_copies.py --dataset ppi_bp --smoke --max-parallel 4   # results/hpo_copies_smoke/
  python run_hpo_copies.py --max-parallel 4 --trials-scale 2          # rerun the same to resume
  python run_hpo_copies.py --collect --trials-scale 2
"""

# Imports
import copies_factored
copies_factored.patch()
import run_hpo as rh
from optuna.distributions import CategoricalDistribution as Cat
from os.path import join
import os

rh.__file__ = os.path.abspath(__file__)  # the trial children run this script, with the patches
rh.RESULTS = join("results", "hpo_copies")
rh.SETUPS = {"colors": {"real": "I_O", "synthetic": "I_O"}}
rh.ORDER = [n for n in rh.ORDER if n in rh.REAL]  # hpo_neuro, hpo_metab, em_user, ppi_bp

# The copies colorings the search picks from (the best ones in run_edge_colors(_copies))
COLORING_CHOICES = ["I_O", "Icopy_Iplain_O", "I_O_C"]

_space, _trial_hps, _build_colored = rh.space, rh.trial_hps, rh.build_colored_relations


def space(name):
    d = _space(name)
    d["aggregation"] = Cat([a for a in rh.AGGREGATIONS if a != "attention"])
    d["coloring"] = Cat(COLORING_CHOICES)
    return d


def trial_hps(name, setup, p):
    """run_hpo's trial_hps with the sampled coloring in place of the setup's fixed one."""
    fixed = rh.SETUPS[setup]["real"]
    rh.SETUPS[setup]["real"] = p["coloring"]
    try:
        return _trial_hps(name, setup, p)
    finally:
        rh.SETUPS[setup]["real"] = fixed


def build_colored_relations(prep, hps):
    """run_trial's warm-up: the factored relations instead of the full outside edges."""
    if copies_factored.needs_factoring(hps):
        return copies_factored.build_factored_relations(prep, hps)
    return _build_colored(prep, hps)


rh.space = space
rh.trial_hps = trial_hps
rh.build_colored_relations = build_colored_relations

if __name__ == "__main__":
    rh.main()
