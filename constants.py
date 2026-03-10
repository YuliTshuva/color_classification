import torch
from os.path import join

# Hyperparameters
COLOR_EMBEDDING_DIM = 8
GNN_EMBEDDING_DIM = 8
GNN_HIDDEN_DIM = 4
K_GNN_LAYERS = 2
GNN_MLP_HIDDEN_DIMS = [4]
ATTENTION_HIDDEN_DIMS = [2]
NUM_ATTENTION_HEADS = 1
GNN_DROPOUT_RATE = 0.7
ATTENTION_DROPOUT_RATE = 0.7
MLP_DROPOUT_RATE = 0.7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data"
DATASET_NAME = "twitch"
EDGES_PATH = join(DATA_DIR, "large_twitch_edges.csv")
NODES_PATH = join(DATA_DIR, "large_twitch_features.csv")
MODELS_DIR = "models"
RESULTS_DIR = "results"
LATIN_LANGUAGES = {
    "EN", "FR", "DE", "ES", "IT", "PT",
    "PL", "NL", "SV", "DA", "FI", "NO",
    "CS", "HU", "TR"
}
