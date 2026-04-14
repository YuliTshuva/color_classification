"""
Yuli Tshuva.
Creating a training script for the model.
By convention the last color is used for testing.
"""

# Imports
from models import *
from utils import *
from sklearn.metrics import roc_auc_score
import json

# Create a random seed for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set the font for the plots
rcParams['font.family'] = "Times New Roman"

# Hyperparameters
EPOCHS = 10000
LR = 0.001
TOLERANCE = 30


def train(n_colors, data):
    """
    Get the data and train the models for color classification.
    :param data: edge_index (2, edges), color_indices (n_vertices,), labels (n_colors,)
    :param n_colors: Amount of colors in the dataset.
    :return: The trained models: GNN, MLP, Color Embedding, Attention Classifier.
    """
    # Initiate the GNN model
    gnn_model = RGCN(in_dim=COLOR_EMBEDDING_DIM,
                     hidden_dim=GNN_HIDDEN_DIM,
                     out_dim=GNN_EMBEDDING_DIM,
                     num_relations=2,
                     num_layers=K_GNN_LAYERS,
                     dropout=GNN_DROPOUT_RATE)

    # Initiate the MLP model for GNN prediction
    mlp_model = MLP(input_dim=GNN_EMBEDDING_DIM,
                    hidden_dims=GNN_MLP_HIDDEN_DIMS,
                    output_dim=1,
                    dropout_rate=MLP_DROPOUT_RATE)

    # Initiate the color embedding model
    color_embedding_model = ColorEmbedding(n_colors=n_colors,
                                           embedding_dim=COLOR_EMBEDDING_DIM)

    # Unpack the data
    edge_index, edge_relation, color_indices, color_labels, test_colors = data

    # Create an array of labels for each node based on the color indices
    node_labels = color_labels[color_indices].float()

    # Set a loss for binary classification
    classification_loss = nn.BCEWithLogitsLoss()

    # Create a training mask for the nodes without the test color index (the last color index)
    training_mask = torch.isin(color_indices, test_colors, invert=True)

    # Set the models to training mode
    gnn_model.train()
    mlp_model.train()
    color_embedding_model.train()

    # Move the models and data to the device
    gnn_model.to(DEVICE)
    mlp_model.to(DEVICE)
    color_embedding_model.to(DEVICE)
    edge_index = edge_index.to(DEVICE)
    color_indices = color_indices.to(DEVICE)
    node_labels = node_labels.to(DEVICE)
    training_mask = training_mask.to(DEVICE)

    # Set optimizers for the models
    optimizer = torch.optim.Adam(list(gnn_model.parameters()) +
                                 list(mlp_model.parameters()) +
                                 list(color_embedding_model.parameters()), lr=LR)

    # Set a variable to store the best loss achieved
    best_loss, unimproved_epochs = torch.inf, 0

    # Set a variable to store the best models
    best_gnn, best_mlp, best_color_embedding, best_attention = None, None, None, None

    # Start the training loop
    for epoch in range(EPOCHS):
        # Embed the color indices for GNN input
        color_embeddings = color_embedding_model(color_indices)

        # Get the GNN output
        gnn_output = gnn_model(color_embeddings, edge_index, edge_relation)

        # Get the MLP output for GNN prediction
        mlp_output = mlp_model(gnn_output).squeeze()

        # Calculate the losses of the training set
        loss_gnn = classification_loss(mlp_output[training_mask], node_labels[training_mask])

        # 1. Calculate the mean per color (Vectorized)
        # Create a zero tensor to store the sums
        sums = torch.zeros(n_colors, gnn_output.size(1), device=DEVICE)
        counts = torch.zeros(n_colors, 1, device=DEVICE)

        # Sum up all embeddings belonging to each color index
        # 'src' is your embeddings, 'index' is your color_indices
        sums.scatter_add_(0, color_indices.unsqueeze(1).expand_as(gnn_output), gnn_output)
        counts.scatter_add_(0, color_indices.unsqueeze(1), torch.ones_like(color_indices).unsqueeze(1).float())

        # Avoid division by zero for colors with no nodes
        counts = torch.clamp(counts, min=1.0)
        color_means = sums / counts

        # 2. Calculate the Variance: Var = E[X^2] - (E[X])^2
        # Expand the means back to the shape of the original nodes
        node_means = color_means[color_indices]
        sq_diff = (gnn_output - node_means) ** 2

        # Sum the squared differences per color
        sq_diff_sums = torch.zeros(n_colors, gnn_output.size(1), device=DEVICE)
        sq_diff_sums.scatter_add_(0, color_indices.unsqueeze(1).expand_as(sq_diff), sq_diff)

        # Calculate final variance per color
        color_variances = sq_diff_sums / counts
        total_variance = color_variances.mean()

        # Combine the losses
        loss = ALPHA * loss_gnn + total_variance

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss < best_loss * 0.99:
            best_loss = loss
            unimproved_epochs = 0
            # Save the best models' state dicts
            best_gnn = gnn_model.state_dict()
            best_mlp = mlp_model.state_dict()
            best_color_embedding = color_embedding_model.state_dict()
        else:
            unimproved_epochs += 1
        if unimproved_epochs >= TOLERANCE:
            break

    # Load the best models
    gnn_model.load_state_dict(best_gnn)
    mlp_model.load_state_dict(best_mlp)
    color_embedding_model.load_state_dict(best_color_embedding)

    # Set the models to evaluation mode
    gnn_model.eval()
    mlp_model.eval()
    color_embedding_model.eval()

    # Predict the outputs for the entire dataset
    with torch.no_grad():
        # Embed the color indices for GNN input
        color_embeddings = color_embedding_model(color_indices)

        # Get the GNN output
        gnn_output = gnn_model(color_embeddings, edge_index, edge_relation)

        # Get the MLP output for GNN prediction
        mlp_output = mlp_model(gnn_output).squeeze()

    # Calculate the predictions and scores
    gnn_scores = torch.sigmoid(mlp_output).squeeze()
    gnn_predictions = (gnn_scores > 0.5).long()

    # Calculate the accuracies for the attention model and the gnn
    node_labels = node_labels.long()
    train_gnn_accuracy = (gnn_predictions[training_mask] == node_labels[training_mask]).float().mean().item()
    test_gnn_accuracy = (gnn_predictions[~training_mask] == node_labels[~training_mask]).float().mean().item()

    # Calculate AUC-ROC for the attention model and the gnn over the training set
    train_gnn_auc = roc_auc_score(node_labels[training_mask].cpu().numpy(), gnn_scores[training_mask].cpu().numpy())
    test_gnn_auc = roc_auc_score(node_labels[~training_mask].cpu().numpy(), gnn_scores[~training_mask].cpu().numpy())

    # Set a dictionary for the results dict
    results_dct = {
        "train_gnn_accuracy": train_gnn_accuracy,
        "test_gnn_accuracy": test_gnn_accuracy,
        "train_gnn_auc": train_gnn_auc,
        "test_gnn_auc": test_gnn_auc,
    }

    return gnn_model, mlp_model, color_embedding_model, results_dct


def main():
    # Load the Twitch dataset
    llm = "MiniLM-L6"
    edge_index, color_indices, labels, split = load_supervised_graph_data(llm=llm)

    # Find n_colors by the length of the labels
    n_colors = len(labels)

    # Set the test colors list
    test_nodes = split == 1
    test_colors = color_indices[test_nodes].unique()

    # Add edges between nodes of the same color
    same_color_edges = []
    for color in range(n_colors):
        color_nodes = (color_indices == color).nonzero(as_tuple=True)[0]
        if len(color_nodes) > 1:
            # Create edges between all pairs of nodes with the same color
            for i in range(len(color_nodes)):
                for j in range(i + 1, len(color_nodes)):
                    same_color_edges.append((color_nodes[i].item(), color_nodes[j].item()))

    # Find the lengths of both edge_index and same_color_edges
    r1, r2 = edge_index.size(1), len(same_color_edges)

    # Convert the same color edges to a tensor and concatenate with the original edge_index
    if same_color_edges:
        same_color_edge_index = torch.tensor(same_color_edges, dtype=torch.long).t().contiguous()
        edge_index = torch.cat([edge_index, same_color_edge_index], dim=1)

    # Create a tensor that indicates which relation each edge belongs to (0 for original edges, 1 for same color edges)
    edge_relation = torch.cat([torch.zeros(r1, dtype=torch.long), torch.ones(r2, dtype=torch.long)], dim=0)

    # Train the model with the current test color
    _, _, _, result_dct = train(n_colors, data=(edge_index, edge_relation, color_indices,
                                                labels, test_colors))
    # Append the results to the results df
    with open(f"{llm}_results.json", "w") as f:
        json.dump(result_dct, f, indent=4)


if __name__ == '__main__':
    main()
