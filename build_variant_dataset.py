"""
Yuli Tshuva.
build_variant_dataset.py - turn one of the anti-collapse encoders from
data/pregnancy_y into a drop-in replacement for data/pregnancy_MiniLM_L6.

The lab's split_data/*_labels.csv is row-for-row identical to the original
train/test_metadata.csv (same post ids, same order, verified), so the node ->
color assignment, the color labels and the train/test split are unchanged: only
the node features and therefore the kNN edges differ. That is exactly the
comparison we want - same task, same colors, new embeddings.

Output layout mirrors data/pregnancy_MiniLM_L6 so every downstream script
(train08/09/10, graph_construction) works by pointing PREGNANCY_DIR at it:

    data/pregnancy_C/
        train_embeddings.npy      variant embeddings, train rows
        test_embeddings.npy       variant embeddings, test rows
        train_metadata.csv        copied (row-aligned with the above)
        test_metadata.csv         copied
        k_20/
            edges.txt             REBUILT from the variant embeddings
            nodes.txt             copied - node -> color map is unchanged
            color_labels.txt      copied
            split.txt             copied

Usage:
  python build_variant_dataset.py --variant C_anticollapse --k 20
  python train08.py --variant rgcn --dataset pregnancy --pregnancy-dir data/pregnancy_C
"""

import argparse
import os
import shutil
from os.path import join

import numpy as np
import torch

from graph_construction import cosine_candidates, _canonical_unique

SOURCE_DIR = join("data", "pregnancy_MiniLM_L6")
EMB_DIR = join("data", "pregnancy_y", "embeddings")


def build(variant, k, out_dir, source_dir=SOURCE_DIR):
    var_dir = join(EMB_DIR, variant)
    train_emb = np.load(join(var_dir, "train_embeddings.npy"))
    test_emb = np.load(join(var_dir, "test_embeddings.npy"))

    # Sanity: the variant must have one row per original node, in the same order
    for split, emb in [("train", train_emb), ("test", test_emb)]:
        n_orig = len(np.load(join(source_dir, f"{split}_embeddings.npy")))
        if len(emb) != n_orig:
            raise ValueError(f"{variant} has {len(emb)} {split} rows, "
                             f"but {source_dir} has {n_orig}")

    os.makedirs(out_dir, exist_ok=True)
    np.save(join(out_dir, "train_embeddings.npy"), train_emb)
    np.save(join(out_dir, "test_embeddings.npy"), test_emb)
    for f in ["train_metadata.csv", "test_metadata.csv"]:
        shutil.copy(join(source_dir, f), join(out_dir, f))

    features = torch.tensor(np.vstack([train_emb, test_emb]), dtype=torch.float)
    n_nodes = features.size(0)
    print(f"{variant}: {n_nodes} nodes, {features.size(1)} dims -> building k={k} graph")

    # Same convention as the original graph: cosine over the raw embeddings,
    # each undirected pair stored once (the RGCN/RGAT layers symmetrise).
    idx, _sim = cosine_candidates(features, k)
    src = torch.arange(n_nodes).unsqueeze(1).expand(n_nodes, k)
    edges, _ = _canonical_unique(src.reshape(-1), idx[:, :k].reshape(-1), n_nodes)

    k_dir = join(out_dir, f"k_{k}")
    os.makedirs(k_dir, exist_ok=True)
    with open(join(k_dir, "edges.txt"), "w") as f:
        f.writelines(f"{int(u)} {int(v)}\n" for u, v in edges.T.tolist())

    # The color structure is identical, so these carry over verbatim
    for f in ["nodes.txt", "color_labels.txt", "split.txt"]:
        shutil.copy(join(source_dir, f"k_{k}", f), join(k_dir, f))

    degrees = torch.bincount(edges.reshape(-1), minlength=n_nodes)
    print(f"  {edges.size(1)} edges | mean degree {degrees.float().mean():.2f} | "
          f"max {int(degrees.max())} | isolated {int((degrees == 0).sum())}")
    print(f"  written to {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Build a pregnancy dataset folder from an anti-collapse variant.")
    parser.add_argument("--variant", type=str, default="C_anticollapse",
                        help="Folder name under data/pregnancy_y/embeddings/")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--out", type=str, default=None,
                        help="Output folder (default: data/pregnancy_<letter>)")
    args = parser.parse_args()

    out = args.out or join("data", "pregnancy_" + args.variant.split("_")[0])
    build(args.variant, args.k, out)


if __name__ == "__main__":
    main()
