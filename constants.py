import torch
from os.path import join

# Hyperparameters
COLOR_EMBEDDING_DIM = 4
GNN_EMBEDDING_DIM = 16
GNN_HIDDEN_DIM = 16
K_GNN_LAYERS = 1
GNN_MLP_HIDDEN_DIMS = [8]
# ATTENTION_HIDDEN_DIMS = [2]
# NUM_ATTENTION_HEADS = 1
GNN_DROPOUT_RATE = 0.3
# ATTENTION_DROPOUT_RATE = 0.7
MLP_DROPOUT_RATE = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data"
DATASET_NAME = "twitch"
SCORES_DIR = "scores"
EDGES_PATH = join(DATA_DIR, "large_twitch_edges.csv")
NODES_PATH = join(DATA_DIR, "large_twitch_features.csv")
MODELS_DIR = "models"
RESULTS_DIR = "results"
LATIN_LANGUAGES = {
    "EN", "FR", "DE", "ES", "IT", "PT",
    "PL", "NL", "SV", "DA", "FI", "NO",
    "CS", "HU", "TR"
}
