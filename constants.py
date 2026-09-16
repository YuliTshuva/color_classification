import os
import torch
from os.path import join

# Hyperparameters
COLOR_EMBEDDING_DIM = 256
GNN_EMBEDDING_DIM = 256
GNN_HIDDEN_DIM = 256
K_GNN_LAYERS = 16
GNN_MLP_HIDDEN_DIMS = [40, 10]
GNN_DROPOUT_RATE = 0.3
NUM_ATTENTION_HEADS = 2
ATTENTION_HIDDEN_DIMS = [32, 8]
ATTENTION_DROPOUT_RATE = 0.2
MLP_DROPOUT_RATE = 0.3
ALPHA, BETA = 1, 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data"
DATASET_NAME = "twitch"
SCORES_DIR = "scores"
EDGES_PATH = join(DATA_DIR, DATASET_NAME, "large_twitch_edges.csv")
NODES_PATH = join(DATA_DIR, DATASET_NAME, "large_twitch_features.csv")
VAE_EDGES = join(DATA_DIR, "VAE_knn", "edges.txt")
VAE_LABELS = join(DATA_DIR, "VAE_knn", "color_labels.txt")
VAE_NODES = join(DATA_DIR, "VAE_knn", "nodes.txt")
MODELS_DIR = "models"
# Which pregnancy embedding set the pipeline runs on. Defaults to the original
# (collapsed) MiniLM embeddings so every existing result reproduces. Point it at
# data/pregnancy_C to run the same experiments on the anti-collapse variant C
# embeddings; the folder layout is identical (train/test_embeddings.npy + k_*/).
# Override with the PREGNANCY_DIR env var, or --pregnancy-dir in train08.
PREGNANCY_DIR = os.environ.get("PREGNANCY_DIR", join(DATA_DIR, "pregnancy_MiniLM_L6"))
# What the GNN reads at its input layer, shared by train08/09/10 (the values are
# listed in train08.NODE_INPUTS). "feat" is the node embeddings, as in every
# experiment so far; the other values drop them and are the featureless
# ablation. Override with the NODE_INPUT env var, or --node-input.
NODE_INPUT = os.environ.get("NODE_INPUT", "feat")
RESULTS_DIR = "results"
RESULTS_DICT = join(RESULTS_DIR, "hp_results.json")
LATIN_LANGUAGES = {
    "EN", "FR", "DE", "ES", "IT", "PT",
    "PL", "NL", "SV", "DA", "FI", "NO",
    "CS", "HU", "TR"
}
