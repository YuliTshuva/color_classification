import torch
from os.path import join

# Hyperparameters
COLOR_EMBEDDING_DIM = 32  # Used to be 512
GNN_EMBEDDING_DIM = 128  # Used to be 256
GNN_HIDDEN_DIM = 128  # Used to be 256
K_GNN_LAYERS = 3  # Used to be 4
GNN_MLP_HIDDEN_DIMS = [32]  # Used to be [128, 64]
ATTENTION_HIDDEN_DIMS = [16]  # Used to be [128, 64]
NUM_ATTENTION_HEADS = 4  # Used to be 8
DROPOUT_RATE = 0.5  # Used to be 0
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
