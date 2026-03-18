import torch
from os.path import join

# Hyperparameters
COLOR_EMBEDDING_DIM = 3
GNN_EMBEDDING_DIM = 36
GNN_HIDDEN_DIM = -1
K_GNN_LAYERS = 1
GNN_MLP_HIDDEN_DIMS = [12]
GNN_DROPOUT_RATE = 0.0
MLP_DROPOUT_RATE = 0.1
ALPHA = 100000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data"
DATASET_NAME = "twitch"
SCORES_DIR = "scores"
EDGES_PATH = join(DATA_DIR, "large_twitch_edges.csv")
NODES_PATH = join(DATA_DIR, "large_twitch_features.csv")
MODELS_DIR = "models"
RESULTS_DIR = "results"
RESULTS_DICT = join(RESULTS_DIR, "hp_results.json")
LATIN_LANGUAGES = {
    "EN", "FR", "DE", "ES", "IT", "PT",
    "PL", "NL", "SV", "DA", "FI", "NO",
    "CS", "HU", "TR"
}
