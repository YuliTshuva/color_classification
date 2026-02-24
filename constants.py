import torch

# Hyperparameters
COLOR_EMBEDDING_DIM = 512
GNN_EMBEDDING_DIM = 256
GNN_HIDDEN_DIM = 256
K_GNN_LAYERS = 5
GNN_MLP_HIDDEN_DIMS = [128, 64]
ATTENTION_HIDDEN_DIMS = [128, 64]
NUM_ATTENTION_HEADS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")