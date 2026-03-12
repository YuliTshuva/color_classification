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
from sklearn.metrics import roc_auc_score
import optuna


# Create a random seed for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set the font for the plots
rcParams['font.family'] = "Times New Roman"

# Hyperparameters
EPOCHS = 10000
LR = 0.001
TOLERANCE = 30


def train(train_id, n_colors, data, hps):
    """
    Get the data and train the models for color classification.
    :param data: edge_index (2, edges), color_indices (n_vertices,), labels (n_colors,)
    :param n_colors: Amount of colors in the dataset.
    :return: The trained models: GNN, MLP, Color Embedding, Attention Classifier.
    """
    # Unpack the hyperparameters
    color_embedding_dim = hps["color_embedding_dim"]
    gnn_embedding_dim = hps["gnn_embedding_dim"]
    gnn_hidden_dim = hps["gnn_hidden_dim"]
    k_gnn_layers = hps["k_gnn_layers"]
    gnn_mlp_hidden_dims = hps["gnn_mlp_hidden_dims"]
    gnn_dropout_rate = hps["gnn_dropout_rate"]
    mlp_dropout_rate = hps["mlp_dropout_rate"]
    alpha = hps["alpha"]

    # Initiate the GNN model
    gnn_model = GNN(in_features=color_embedding_dim,
                    hidden_dim=gnn_hidden_dim,
                    out_features=gnn_embedding_dim,
                    K=k_gnn_layers,
                    dropout_rate=gnn_dropout_rate)

    # Initiate the MLP model for GNN prediction
    mlp_model = MLP(input_dim=gnn_embedding_dim,
                    hidden_dims=gnn_mlp_hidden_dims,
                    output_dim=1,
                    dropout_rate=mlp_dropout_rate)

    # Initiate the color embedding model
    color_embedding_model = ColorEmbedding(n_colors=n_colors,
                                           embedding_dim=color_embedding_dim)

    # Unpack the data
    edge_index, color_indices, color_labels, test_colors = data

    # Create an array of labels for each node based on the color indices
    node_labels = color_labels[color_indices].float()

    # Set a loss for binary classification
    classification_loss = nn.BCEWithLogitsLoss()

    # Set an enumerator for the colors
    colors_enumerator = torch.arange(0, n_colors).long()

    # Create a training mask for the nodes without the test color index (the last color index)
    training_mask = torch.isin(color_indices, test_colors, invert=True)
    training_colors = torch.isin(colors_enumerator, test_colors, invert=True)

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
    color_labels = color_labels.to(DEVICE)
    node_labels = node_labels.to(DEVICE)
    # colors_enumerator = colors_enumerator.to(DEVICE)
    training_mask = training_mask.to(DEVICE)

    # Set optimizers for the models
    optimizer = torch.optim.Adam(list(gnn_model.parameters()) +
                                 list(mlp_model.parameters()) +
                                 list(color_embedding_model.parameters()), lr=LR)

    # Set a list to store the losses for plotting
    losses_gnn, variances, losses_total = [], [], []

    # Set a variable to store the best loss achieved
    best_loss, unimproved_epochs = torch.inf, 0

    # Start the training loop
    for epoch in tqdm(range(EPOCHS), desc="Training", total=EPOCHS):
        # Embed the color indices for GNN input
        color_embeddings = color_embedding_model(color_indices)

        # Get the GNN output
        gnn_output = gnn_model(color_embeddings, edge_index)

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
        loss = alpha * loss_gnn + total_variance

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Append the losses and variance for plotting
        losses_gnn.append(loss_gnn.detach().cpu().item())
        variances.append(total_variance.detach().cpu().item())
        losses_total.append(loss.detach().cpu().item())

        if loss < best_loss * 0.99:
            best_loss = loss
            unimproved_epochs = 0
            # Save the best models
            os.makedirs(MODELS_DIR, exist_ok=True)
            torch.save(gnn_model.state_dict(), join(MODELS_DIR, f"gnn_model_{train_id}.pth"))
            torch.save(mlp_model.state_dict(), join(MODELS_DIR, f"mlp_model_{train_id}.pth"))
            torch.save(color_embedding_model.state_dict(), join(MODELS_DIR, f"color_embedding_model_{train_id}.pth"))
        else:
            unimproved_epochs += 1
        if unimproved_epochs >= TOLERANCE:
            print(f"Early stopping at epoch {epoch} with best loss {best_loss:.4f}")
            break

    # Load the best models
    gnn_model.load_state_dict(torch.load(join(MODELS_DIR, f"gnn_model_{train_id}.pth")))
    mlp_model.load_state_dict(torch.load(join(MODELS_DIR, f"mlp_model_{train_id}.pth")))
    color_embedding_model.load_state_dict(torch.load(join(MODELS_DIR, f"color_embedding_model_{train_id}.pth")))

    # Set the models to evaluation mode
    gnn_model.eval()
    mlp_model.eval()
    color_embedding_model.eval()

    # Predict the outputs for the entire dataset
    with torch.no_grad():
        # Embed the color indices for GNN input
        color_embeddings = color_embedding_model(color_indices)

        # Get the GNN output
        gnn_output = gnn_model(color_embeddings, edge_index)

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

    # Set a dictionary for the results dict
    results_dct = {
        "train_gnn_accuracy": train_gnn_accuracy,
        "test_gnn_accuracy": test_gnn_accuracy,
        "test_label": color_labels[~training_colors].cpu().item(),
        "train_gnn_auc": train_gnn_auc,
    }

    # Plot the losses and variance
    plt.figure(figsize=(7, 6))
    plt.plot(losses_gnn, label=f'GNN Loss', color="dodgerblue")
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

    return gnn_model, mlp_model, color_embedding_model, results_dct


def objective(trial):
    # 1. Define the HP space using your grid
    config = {
        "color_embedding_dim": trial.suggest_categorical("color_embedding_dim", [2, 3, 4, 5, 6, 8, 10]),
        "gnn_embedding_dim": trial.suggest_categorical("gnn_embedding_dim", [16, 24, 32, 48, 64, 100]),
        "gnn_hidden_dim": trial.suggest_categorical("gnn_hidden_dim", [8, 12, 16, 24, 32, 48]),
        "k_gnn_layers": trial.suggest_categorical("k_gnn_layers", [1, 2, 3]),
        "gnn_mlp_hidden_dims": trial.suggest_categorical("gnn_mlp_hidden_dims", [[2, 2], [4], [8], [10]]),
        "gnn_dropout_rate": trial.suggest_categorical("gnn_dropout_rate", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        "mlp_dropout_rate": trial.suggest_categorical("mlp_dropout_rate", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        "alpha": trial.suggest_categorical("alpha", [1e-4, 1e-3, 1e-2, 0.1, 1, 1e1, 1e2, 1e3]),
    }

    # Load data (Better to do this outside objective if it's heavy)
    edge_index, color_indices, labels = load_twitch_data()
    n_colors = len(labels)

    test_accs, test_labels = [], []

    # 2. Run your evaluation loop
    for i in range(n_colors):
        test_colors = torch.tensor([i])
        # Pass the 'config' dictionary into your train function
        _, _, _, result_dct = train(i, n_colors, data=(edge_index, color_indices, labels, test_colors), **config)

        test_accs.append(result_dct["test_gnn_accuracy"])
        test_labels.append(result_dct["test_label"])

    # 3. Calculate the metric to maximize
    return roc_auc_score(test_labels, test_accs)


def main():
    # Create a study to MAXIMIZE AUC
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)  # Set trials based on your time budget

    print("Best AUC:", study.best_value)
    print("Best params:", study.best_params)

    # Save the best hyperparameters to a file
    with open(join(MODELS_DIR, "best_hyperparameters.txt"), "w") as f:
        f.write(f"Best AUC: {study.best_value}\n")
        f.write("Best Hyperparameters:\n")
        for key, value in study.best_params.items():
            f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    main()
