"""
Yuli Tshuva.
Creating a training script for the model.
By convention the last color is used for testing.
"""

# Imports
import os
from models import *
from utils import *
from tqdm.auto import tqdm
import pandas as pd

# Set the font for the plots
rcParams['font.family'] = "Times New Roman"

# Hyperparameters
EPOCHS = 10000
ALPHA, BETA, GAMMA = 1, 1, 1
LR = 0.001
TOLERANCE = 70

# Constants
FREEZE_COLOR_EMBEDDING = False


def train(n_classes, n_colors, data):
    """
    Get the data and train the models for color classification.
    :param data: edge_index (2, edges), color_indices (n_vertices,), labels (n_colors,)
    :param n_classes: Amount of classes to classify the colors into.
    :param n_colors: Amount of colors in the dataset.
    :return: The trained models: GNN, MLP, Color Embedding, Attention Classifier.
    """
    # Initiate the GNN model
    gnn_model = GNN(in_features=COLOR_EMBEDDING_DIM,
                    hidden_dim=GNN_HIDDEN_DIM,
                    out_features=GNN_EMBEDDING_DIM,
                    K=K_GNN_LAYERS)

    # Initiate the MLP model for GNN prediction
    mlp_model = MLP(input_dim=GNN_EMBEDDING_DIM,
                    hidden_dims=GNN_MLP_HIDDEN_DIMS,
                    output_dim=n_classes)

    # Initiate the color embedding model
    color_embedding_model = ColorEmbedding(n_colors=n_colors,
                                           embedding_dim=COLOR_EMBEDDING_DIM)

    # Initiate the attention color classification model
    attention_model = AttentionColorClassifier(dim=COLOR_EMBEDDING_DIM,
                                               num_heads=NUM_ATTENTION_HEADS,
                                               num_classes=n_classes)

    # Unpack the data
    edge_index, color_indices, color_labels, test_colors = data

    # Create an array of labels for each node based on the color indices
    node_labels = color_labels[color_indices]

    # Set a loss for classification
    classification_loss = nn.CrossEntropyLoss()

    # Set an enumerator for the colors
    colors_enumerator = torch.arange(0, n_colors).long()

    # Create a training mask for the nodes without the test color index (the last color index)
    training_mask = torch.isin(color_indices, test_colors, invert=True)
    training_colors = torch.isin(colors_enumerator, test_colors, invert=True)

    # Set the models to training mode
    gnn_model.train()
    mlp_model.train()
    color_embedding_model.train()
    attention_model.train()

    # Move the models and data to the device
    gnn_model.to(DEVICE)
    mlp_model.to(DEVICE)
    color_embedding_model.to(DEVICE)
    attention_model.to(DEVICE)
    edge_index = edge_index.to(DEVICE)
    color_indices = color_indices.to(DEVICE)
    color_labels = color_labels.to(DEVICE)
    node_labels = node_labels.to(DEVICE)
    colors_enumerator = colors_enumerator.to(DEVICE)
    training_mask = training_mask.to(DEVICE)

    # Set optimizers for the models
    optimizer = torch.optim.Adam(list(gnn_model.parameters()) +
                                 list(mlp_model.parameters()) +
                                 list(color_embedding_model.parameters()) +
                                 list(attention_model.parameters()), lr=LR)

    # Set a list to store the losses for plotting
    losses_gnn, losses_attention, variances, losses_total = [], [], [], []

    # Set a variable to store the best loss achieved
    best_loss, unimproved_epochs = torch.inf, 0

    # Start the training loop
    for epoch in tqdm(range(EPOCHS), desc="Training", total=EPOCHS):
        # Embed the color indices for GNN input
        color_embeddings = color_embedding_model(color_indices)

        # Get the GNN output
        gnn_output = gnn_model(color_embeddings, edge_index)

        # Get the MLP output for GNN prediction
        mlp_output = mlp_model(gnn_output)

        # Embed colors enumerator for attention model
        colors_emb = color_embedding_model(colors_enumerator)
        # Get the attention model output
        attention_output = attention_model(colors_emb)

        # Calculate the losses of the training set
        loss_gnn = classification_loss(mlp_output[training_mask], node_labels[training_mask])
        loss_attention = classification_loss(attention_output[training_colors], color_labels[training_colors])

        # Get the variance of the GNN embeddings for all the nodes in the same color
        color_variances = []
        # Iterate over each color
        for color in range(n_colors):
            # Find the color's nodes
            color_nodes = (color_indices == color)
            # Check if there are nodes for this color
            if color_nodes.sum() < 1:
                raise Exception(f"Color {color} with no nodes found in the graph.")
            elif color_nodes.sum() == 1:
                raise Exception(f"Color {color} with one node found in the graph.")
            # Get the GNN embeddings for the color's nodes
            color_embeddings_for_color = gnn_output[color_nodes]
            # Find the variance of the GNN embeddings for the color's nodes
            variance = torch.var(color_embeddings_for_color, dim=0).mean()
            color_variances.append(variance)
        # Sum the variances for all colors
        total_variance = torch.mean(torch.stack(color_variances))

        # Combine the losses
        loss = ALPHA * loss_gnn + BETA * loss_attention + GAMMA * total_variance

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Append the losses and variance for plotting
        losses_gnn.append(loss_gnn.detach().cpu().item())
        losses_attention.append(loss_attention.detach().cpu().item())
        variances.append(total_variance.detach().cpu().item())
        losses_total.append(loss.detach().cpu().item())

        if loss < best_loss * 0.99:
            best_loss = loss
            unimproved_epochs = 0
        else:
            unimproved_epochs += 1
        if unimproved_epochs >= TOLERANCE:
            print(f"Early stopping at epoch {epoch} with best loss {best_loss:.4f}")
            break

    # Calculate the accuracy for GNN and attention models on both training and testing sets
    gnn_predictions = mlp_output.argmax(dim=1)
    attention_predictions = attention_output.argmax(dim=1)
    train_gnn_accuracy = (gnn_predictions[training_mask] == node_labels[training_mask]).float().mean().item()
    train_attention_accuracy = (
            attention_predictions[training_colors] == color_labels[training_colors]).float().mean().item()
    test_gnn_accuracy = (gnn_predictions[~training_mask] == node_labels[~training_mask]).float().mean().item()
    test_attention_accuracy = (
            attention_predictions[~training_colors] == color_labels[~training_colors]).float().mean().item()
    print(f"GNN Accuracy on Test Set: {test_gnn_accuracy:.4f}")
    print(f"Attention Accuracy on Test Set: {test_attention_accuracy:.4f}")
    print(f"GNN Accuracy on Training Set: {train_gnn_accuracy:.4f}")
    print(f"Attention Accuracy on Training Set: {train_attention_accuracy:.4f}")

    # Check AUC
    attention_scores = attention_output.softmax(dim=1)[:, 1].detach().cpu().numpy()

    # Set a dictionary for the results dict
    results_dct = {
        "train_gnn_accuracy": train_gnn_accuracy,
        "train_attention_accuracy": train_attention_accuracy,
        "test_gnn_accuracy": test_gnn_accuracy,
        "test_attention_accuracy": test_attention_accuracy,
        "attention_score": attention_scores[~training_colors].item(),
        "test_labels": color_labels[~training_colors].cpu().numpy()
    }

    # Plot the losses and variance
    plt.figure(figsize=(7, 6))
    plt.plot(losses_gnn, label=f'GNN Loss', color="dodgerblue")
    plt.plot(losses_attention, label=f'Attention Loss', color="hotpink")
    plt.plot(variances, label='Variance', color="turquoise")
    plt.plot(losses_total, label='Total Loss', color="salmon")
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss Value', fontsize=16)
    plt.title('Training Losses', fontsize=22)
    plt.legend(fontsize=14)
    plt.grid()
    plt.tight_layout()
    plt.savefig(join("plots", f"training_losses_{'_'.join([str(element.item()) for element in test_colors])}.png"))
    plt.close()

    return gnn_model, mlp_model, color_embedding_model, attention_model, results_dct


def main():
    # Set the n_classes to 2 as devora said
    n_classes = 2

    # Load the Twitch dataset
    edge_index, color_indices, labels = load_twitch_data()

    # Find n_colors by the length of the labels
    n_colors = len(labels)

    # Set a results df
    results_df = pd.DataFrame(columns=["train_gnn_accuracy", "train_attention_accuracy",
                                       "test_gnn_accuracy", "test_attention_accuracy",
                                       "attention_score", "test_labels"])

    # Train the model
    for i in tqdm(range(n_colors), desc="Training for each color", total=n_colors):
        # Define test labels
        test_colors = torch.tensor([i])
        # Train the model with the current test color
        _, _, _, _, result_dct = train(n_classes, n_colors, data=(edge_index, color_indices,
                                                                  labels, test_colors))
        # Append the results to the results df
        results_df.loc[i] = result_dct

        # Save the results
        os.makedirs(RESULTS_DIR, exist_ok=True)
        results_df.to_csv(join(RESULTS_DIR, "results.csv"), index=True)


if __name__ == '__main__':
    main()
