"""
Yuli Tshuva.
Creating a training script for the model.
"""

# Imports
from os.path import join
import matplotlib.pyplot as plt
from matplotlib import rcParams
from models import *
from constants import *
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score

# Set the font for the plots
rcParams['font.family'] = "Times New Roman"

# Hyperparameters
EPOCHS = 1000
ALPHA, BETA, GAMMA = 1, 1, 1
LR = 0.001
TOLERANCE = 70

# Constants
FREEZE_COLOR_EMBEDDING = False


def train(n_classes, n_colors, data):
    """

    :param data: edge_index (2, edges), color_indices (n_vertices,), labels (n_colors,)
    :param n_classes: Amount of classes to classify the colors into.
    :param n_colors: Amount of colors in the dataset.
    :return:
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
    edge_index, color_indices, color_labels = data

    # Create an array of labels for each node based on the color indices
    node_labels = color_labels[color_indices]

    # Set a loss for classification
    classification_loss = nn.CrossEntropyLoss()

    # Set an enumerator for the colors
    colors_enumerator = torch.arange(0, n_colors).long()

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

    # Set optimizers for the models
    optimizer = torch.optim.Adam(list(gnn_model.parameters()) +
                                 list(mlp_model.parameters()) +
                                 # list(color_embedding_model.parameters()) +
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
        colors = color_embedding_model(colors_enumerator)
        # Get the attention model output
        attention_output = attention_model(colors)

        # Calculate the losses
        loss_gnn = classification_loss(mlp_output, node_labels)
        loss_attention = classification_loss(attention_output, color_labels)

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

    # Calculate the final accuracy for GNN and attention models
    gnn_predictions = mlp_output.argmax(dim=1)
    attention_predictions = attention_output.argmax(dim=1)
    gnn_accuracy = (gnn_predictions == node_labels).float().mean().item()
    attention_accuracy = (attention_predictions == color_labels).float().mean().item()
    print(f"Final GNN Accuracy: {gnn_accuracy:.4f}")
    print(f"Final Attention Accuracy: {attention_accuracy:.4f}")

    # Check AUC
    gnn_scores = mlp_output.softmax(dim=1)[:, 1].detach().cpu().numpy()
    attention_scores = attention_output.softmax(dim=1)[:, 1].detach().cpu().numpy()
    node_labels_np = node_labels.detach().cpu().numpy()
    color_labels_np = color_labels.detach().cpu().numpy()
    gnn_auc = roc_auc_score(node_labels_np, gnn_scores)
    attention_auc = roc_auc_score(color_labels_np, attention_scores)
    print(f"GNN AUC: {gnn_auc:.4f}")
    print(f"Attention AUC: {attention_auc:.4f}")


    # Plot the losses and variance
    plt.figure(figsize=(7, 6))
    plt.plot(losses_gnn, label=f'GNN Loss (acc: {gnn_accuracy:.4f} AUC: {gnn_auc:.4f})', color="dodgerblue")
    plt.plot(losses_attention, label=f'Attention Loss (acc: {attention_accuracy:.4f} AUC: {attention_auc:.4f})', color="hotpink")
    plt.plot(variances, label='Variance', color="turquoise")
    plt.plot(losses_total, label='Total Loss', color="salmon")
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss Value', fontsize=16)
    plt.title('Training Losses', fontsize=22)
    plt.legend(fontsize=14)
    plt.grid()
    plt.tight_layout()
    plt.savefig(join("plots", "training_losses.png"))
    plt.show()


def generate_random_data(n_vertices, n_colors, n_classes):
    # Create edges between vertices
    edge_index = torch.randint(0, n_vertices, (2, 15 * n_vertices))  # 5000 random edges
    # Create random color indices for each vertex
    color_indices = torch.randint(0, n_colors, (n_vertices,))
    # Create random labels for each vertex
    labels = torch.randint(0, n_classes, (n_colors,))
    return edge_index, color_indices, labels


def main():
    n_classes = 2
    n_vertices = 1000
    n_colors = n_vertices // 10

    # Generate random data
    edge_index, color_indices, labels = generate_random_data(n_vertices, n_colors, n_classes)

    # Train the model
    train(n_classes, n_colors, data=(edge_index, color_indices, labels))


if __name__ == '__main__':
    main()
