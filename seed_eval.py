"""
Yuli Tshuva.
seed_eval.py - Robustness check for the exp08 results: retrain the best
configuration of each variant (selected by validation AUC from its study)
over several seeds, to put error bars on the reported AUCs. The train/val
color split stays fixed (seed 42); only model init / dropout vary.

Usage (server):
  python seed_eval.py                       # all 6 variants, 5 seeds each
  python seed_eval.py --variants rgat rgcn --seeds 10

Results are appended to results/exp08/<dataset>/seed_results.csv.
"""

# Imports
from train08 import (train, load_pregnancy_node_features, load_disease_node_features,
                     FEATURE_VARIANTS, VARIANTS, DISEASES, EPOCHS)
from utils import load_pregnancy_data, load_disease_data
from os.path import join
import argparse
import csv
import json
import os
import pandas as pd


def best_hps_of(dataset, variant):
    """Pick the study's best config (by mean validation AUC) from its trial CSV."""
    df = pd.read_csv(join("results", "exp08", dataset, variant, "trial_results.csv"))
    df["val_score"] = (df["val_gnn_auc"] + df["val_color_auc"]) / 2
    per_config = df.groupby("hps")["val_score"].mean()
    return json.loads(per_config.idxmax())


def main():
    parser = argparse.ArgumentParser(description="Multi-seed evaluation of the best exp08 configs.")
    parser.add_argument("--dataset", type=str, default="pregnancy", choices=["diseases", "pregnancy"])
    parser.add_argument("--variants", type=str, nargs="+", default=VARIANTS)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    csv_path = join("results", "exp08", args.dataset, "seed_results.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ["variant", "seed", "disease", "train_gnn_auc", "val_gnn_auc", "test_gnn_auc",
                  "train_color_auc", "val_color_auc", "test_color_auc", "hps"]

    dataset_names = DISEASES if args.dataset == "diseases" else ["pregnancy"]

    # Load data (and features) once
    data_cache, feat_cache = {}, {}

    for variant in args.variants:
        hps = best_hps_of(args.dataset, variant)
        print(f"== {variant}: best hps {hps}")
        k = hps["k"]

        for name in dataset_names:
            if (name, k) not in data_cache:
                if args.dataset == "diseases":
                    edge_index, color_indices, labels, split = load_disease_data(disease=name, k=k)
                else:
                    edge_index, color_indices, labels, split = load_pregnancy_data(k=k)
                test_colors = color_indices[split == 1].unique()
                data_cache[(name, k)] = (edge_index, color_indices, labels, test_colors, split)
            if variant in FEATURE_VARIANTS and name not in feat_cache:
                feat_cache[name] = (load_disease_node_features(name) if args.dataset == "diseases"
                                    else load_pregnancy_node_features())

            edge_index, color_indices, labels, test_colors, split = data_cache[(name, k)]
            n_colors = len(labels)

            for seed in range(args.seeds):
                (test_gnn_auc, val_gnn_auc, train_gnn_auc,
                 test_color_auc, val_color_auc, train_color_auc) = train(
                    variant, n_colors,
                    data=(edge_index, color_indices, labels, test_colors, split),
                    hps=hps, features=feat_cache.get(name), epochs=args.epochs, seed=seed)

                write_header = not os.path.exists(csv_path)
                with open(csv_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if write_header:
                        writer.writeheader()
                    writer.writerow({"variant": variant, "seed": seed, "disease": name,
                                     "train_gnn_auc": round(train_gnn_auc, 4),
                                     "val_gnn_auc": round(val_gnn_auc, 4),
                                     "test_gnn_auc": round(test_gnn_auc, 4),
                                     "train_color_auc": round(train_color_auc, 4),
                                     "val_color_auc": round(val_color_auc, 4),
                                     "test_color_auc": round(test_color_auc, 4),
                                     "hps": json.dumps(hps)})
                print(f"{variant} seed {seed} [{name}]: "
                      f"test gnn {test_gnn_auc:.4f}, test color {test_color_auc:.4f}")

    print(f"Done. Results in {csv_path}")


if __name__ == "__main__":
    main()
