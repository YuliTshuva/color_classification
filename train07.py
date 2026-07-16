"""
Yuli Tshuva.
Creating a training script for the model.
Same as train06.py but for the pregnancy_MiniLM_L6 dataset instead of tamars_datasets.
By convention the last color is used for testing.
"""

# Imports
from models import *
from utils import *
from sklearn.metrics import roc_auc_score
import ast
import csv
import json
import optuna
import os
import random

# Create a random seed for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set the font for the plots
rcParams['font.family'] = "Times New Roman"

# Hyperparameters
EPOCHS = 10000
LR = 0.001
TOLERANCE = 30


def train(n_colors, data, hps, directed=False, plot=True):
    """
    Get the data and train the models for color classification.
    :param data: edge_index (2, edges), color_indices (n_vertices,), labels (n_colors,)
    :param n_colors: Amount of colors in the dataset.
    :param plot: Whether to plot train/val losses after training.
    :return: The trained models: GNN, MLP, Color Embedding, Attention Classifier.
    """
    # Read the HPs from the file
    color_embedding_dim = hps["color_embedding_dim"]
    gnn_embedding_dim = hps["gnn_embedding_dim"]
    gnn_hidden_dim = hps["gnn_hidden_dim"]
    k_gnn_layers = hps["k_gnn_layers"]
    gnn_dropout_rate = hps["gnn_dropout_rate"]
    gnn_mlp_hidden_dims = [hps["gnn_mlp_hidden_dims"]]
    mlp_dropout_rate = hps["mlp_dropout_rate"]
    weight_decay = hps["weight_decay"]
    attention_hidden_dims = [hps["attention_hidden_dims"]]
    attention_dropout_rate = hps["attention_dropout_rate"]
    n_layers = hps["n_layers"]
    model = hps["model"]

    # Initiate the GNN model
    if model == "GCN":
        gnn_model = GCN(in_features=color_embedding_dim,
                        hidden_dim=gnn_hidden_dim,
                        out_features=gnn_embedding_dim,
                        K=k_gnn_layers,
                        dropout_rate=gnn_dropout_rate)
    elif model == "GAT":
        gnn_model = GAT(in_features=color_embedding_dim,
                        hidden_dim=gnn_hidden_dim,
                        out_features=gnn_embedding_dim,
                        K=k_gnn_layers,
                        dropout_rate=gnn_dropout_rate,
                        heads=4)
    elif model == "GIN":
        gnn_model = GIN(in_features=color_embedding_dim,
                        hidden_dim=gnn_hidden_dim,
                        out_features=gnn_embedding_dim,
                        K=k_gnn_layers,
                        dropout_rate=gnn_dropout_rate)
    elif model == "SAGE":
        gnn_model = GraphSAGE(in_features=color_embedding_dim,
                              hidden_dim=gnn_hidden_dim,
                              out_features=gnn_embedding_dim,
                              K=k_gnn_layers,
                              dropout_rate=gnn_dropout_rate)
    else:
        raise ValueError(f"Invalid model type: {model}")

    # Initiate the MLP model for GNN prediction
    mlp_model = MLP(input_dim=gnn_embedding_dim,
                    hidden_dims=gnn_mlp_hidden_dims,
                    output_dim=1,
                    dropout_rate=mlp_dropout_rate)

    # Initiate the color embedding model
    color_embedding_model = ColorEmbedding(n_colors=n_colors,
                                           embedding_dim=color_embedding_dim)

    # Initiate the color classifier model
    color_classifier_model = AttentionClassifier(dim=color_embedding_dim,
                                                 num_heads=2,
                                                 num_classes=1,
                                                 attention_hidden_dims=attention_hidden_dims,
                                                 attention_dropout_rate=attention_dropout_rate,
                                                 n_layers=n_layers)

    # Unpack the data
    edge_index, color_indices, color_labels, test_colors = data

    # Create an array of labels for each node based on the color indices
    color_labels = color_labels.float()
    node_labels = color_labels[color_indices].float()

    # Set a loss for binary classification
    classification_loss = nn.BCEWithLogitsLoss()

    # Split non-test colors 80/20 into train and validation
    all_colors = torch.arange(n_colors)
    train_val_colors = all_colors[~torch.isin(all_colors, test_colors)]
    n_val = max(1, int(len(train_val_colors) * 0.2))
    perm = torch.randperm(len(train_val_colors), generator=torch.Generator().manual_seed(42))
    val_colors = train_val_colors[perm[:n_val]]
    train_colors = train_val_colors[perm[n_val:]]
    # Set the masks
    training_mask = torch.isin(color_indices, train_colors)
    val_mask = torch.isin(color_indices, val_colors)
    test_mask = torch.isin(color_indices, test_colors)

    # Set the models to training mode
    gnn_model.train()
    mlp_model.train()
    color_embedding_model.train()
    color_classifier_model.train()

    # Move the models and data to the device
    gnn_model.to(DEVICE)
    mlp_model.to(DEVICE)
    color_embedding_model.to(DEVICE)
    color_classifier_model.to(DEVICE)
    edge_index = edge_index.to(DEVICE)
    color_indices = color_indices.to(DEVICE)
    color_labels = color_labels.to(DEVICE)
    node_labels = node_labels.to(DEVICE)
    training_mask = training_mask.to(DEVICE)
    val_mask = val_mask.to(DEVICE)
    test_mask = test_mask.to(DEVICE)
    all_color_ids = all_colors.to(DEVICE)

    # Set optimizers for the models (AdamW = decoupled weight decay / L2 regularization)
    optimizer = torch.optim.AdamW(list(gnn_model.parameters()) +
                                  list(mlp_model.parameters()) +
                                  list(color_embedding_model.parameters()) +
                                  list(color_classifier_model.parameters()),
                                  lr=LR, weight_decay=weight_decay)

    # Set a variable to store the best validation loss achieved
    best_val_loss, unimproved_epochs = torch.inf, 0

    # Set a variable to store the best models
    best_gnn, best_mlp, best_color_embedding, best_color_classifier = None, None, None, None

    # Track losses per epoch
    train_losses, val_losses = [], []

    # Start the training loop
    for epoch in range(EPOCHS):
        # Embed the color indices for GNN input
        color_embeddings = color_embedding_model(color_indices)

        # Get the GNN output
        gnn_output = gnn_model(color_embeddings, edge_index)

        # Get the MLP output for GNN prediction
        mlp_output = mlp_model(gnn_output).squeeze()

        # Predict the color classes for the color classification task (one token per color)
        color_classification_output = color_classifier_model(color_embedding_model(all_color_ids)).squeeze()

        # Calculate the losses of the training set
        loss_gnn = classification_loss(mlp_output[training_mask], node_labels[training_mask])
        loss_att = classification_loss(color_classification_output[train_colors], color_labels[train_colors])
        loss = loss_gnn + loss_att

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute validation loss (no gradients needed)
        with torch.no_grad():
            val_loss_gnn = classification_loss(mlp_output[val_mask].detach(), node_labels[val_mask])
            val_loss_att = classification_loss(color_classification_output[val_colors].detach(),
                                               color_labels[val_colors])
            val_loss = val_loss_gnn + val_loss_att
            val_loss = val_loss.item()

        # Plot the losses
        train_losses.append(loss.detach().item())
        val_losses.append(val_loss)

        # Early stopping based on validation loss
        if val_loss < best_val_loss * 0.99:
            best_val_loss = val_loss
            unimproved_epochs = 0
            # Save the best models' state dicts
            best_gnn = gnn_model.state_dict()
            best_mlp = mlp_model.state_dict()
            best_color_embedding = color_embedding_model.state_dict()
            best_color_classifier = color_classifier_model.state_dict()
        else:
            unimproved_epochs += 1
        if unimproved_epochs >= TOLERANCE:
            break

    # Load the best models
    gnn_model.load_state_dict(best_gnn)
    mlp_model.load_state_dict(best_mlp)
    color_embedding_model.load_state_dict(best_color_embedding)
    color_classifier_model.load_state_dict(best_color_classifier)

    # Set the models to evaluation mode
    gnn_model.eval()
    mlp_model.eval()
    color_embedding_model.eval()
    color_classifier_model.eval()

    # Predict the outputs for the entire dataset
    with torch.no_grad():
        # Embed the color indices for GNN input
        color_embeddings = color_embedding_model(color_indices)

        # Get the GNN output
        gnn_output = gnn_model(color_embeddings, edge_index)

        # Get the MLP output for GNN prediction
        mlp_output = mlp_model(gnn_output).squeeze()

        # Predict the color classes for the color classification task (one token per color)
        color_classification_output = color_classifier_model(color_embedding_model(all_color_ids)).squeeze()

    # Calculate the predictions and scores
    gnn_scores = torch.sigmoid(mlp_output).squeeze()
    color_scores = torch.sigmoid(color_classification_output).squeeze()

    # Calculate AUC-ROC over the train, validation, and test sets
    node_labels = node_labels.long()
    train_gnn_auc = roc_auc_score(node_labels[training_mask].cpu().numpy(), gnn_scores[training_mask].cpu().numpy())
    val_gnn_auc = roc_auc_score(node_labels[val_mask].cpu().numpy(), gnn_scores[val_mask].cpu().numpy())
    test_gnn_auc = roc_auc_score(node_labels[test_mask].cpu().numpy(), gnn_scores[test_mask].cpu().numpy())
    train_color_auc = roc_auc_score(color_labels[train_colors].cpu().numpy(), color_scores[train_colors].cpu().numpy())
    val_color_auc = roc_auc_score(color_labels[val_colors].cpu().numpy(), color_scores[val_colors].cpu().numpy())
    test_color_auc = roc_auc_score(color_labels[test_colors].cpu().numpy(), color_scores[test_colors].cpu().numpy())

    return test_gnn_auc, val_gnn_auc, train_gnn_auc, test_color_auc, val_color_auc, train_color_auc


def analyze_results():
    """
    Select the best trial by *validation* AUC (the metric the Optuna study
    optimizes) and report that trial's *test* AUC as the unbiased estimate.

    The hyperparameters are recovered by matching the trial's validation
    score back to hp_results_pregnancy.json, whose stored value is exactly
    that quantity.
    """
    # Load the per-trial results logged during the study
    csv_path = join("results", "trial_results_pregnancy.csv")
    df = pd.read_csv(csv_path)

    # Mean (gnn, color) score per trial
    df["val_score"] = (df["val_gnn_auc"] + df["val_color_auc"]) / 2
    df["test_score"] = (df["test_gnn_auc"] + df["test_color_auc"]) / 2
    per_trial = df.groupby("trial").mean(numeric_only=True)

    # Pick the trial with the best validation score
    best_trial = per_trial["val_score"].idxmax()
    best = per_trial.loc[best_trial]

    # Recover the hyperparameters by matching the val score to hp_results_pregnancy.json.
    # CSV AUCs are rounded to 4 decimals while the json stores full precision,
    # so take the closest entry (within a rounding-error tolerance).
    best_hps = None
    hp_path = join("results", "hp_results_pregnancy.json")
    if os.path.exists(hp_path):
        with open(hp_path, "r") as f:
            hp_tried = json.load(f)
        if hp_tried:
            hps_str, score = min(hp_tried.items(),
                                 key=lambda kv: abs(kv[1] - best["val_score"]))
            if abs(score - best["val_score"]) < 1e-3:
                best_hps = ast.literal_eval(hps_str)

    print(f"Best trial (by validation AUC): {best_trial}")
    print(f"Best hyperparameters: {best_hps}")
    print(f"  Validation AUC: {best['val_score']:.4f} "
          f"(gnn {best['val_gnn_auc']:.4f}, color {best['val_color_auc']:.4f})")
    print(f"  Test AUC:       {best['test_score']:.4f} "
          f"(gnn {best['test_gnn_auc']:.4f}, color {best['test_color_auc']:.4f})")
    print(f"  Train AUC:      "
          f"(gnn {best['train_gnn_auc']:.4f}, color {best['train_color_auc']:.4f})")

    return best_hps


def hyper_parameters_optimization():
    # Set the hyperparameter ranges for the grid search
    color_embedding_dim_range = [12, 16, 20, 24]
    gnn_embedding_dim_range = [100, 128, 156]
    gnn_hidden_dim_range = [100, 128, 156]
    k_gnn_layers_range = [1, 2, 3]
    gnn_dropout_rate_range = [0.0, 0.1, 0.2]
    gnn_mlp_hidden_dims_range = [50, 64, 80]
    mlp_dropout_rate_range = [0.0, 0.1, 0.2]
    weight_decay_range = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
    attention_hidden_dims_range = [4, 6, 8]
    attention_dropout_rate_range = [0.0, 0.1, 0.2]
    n_layers_range = [1, 2, 3, 4]
    # k_range = [3, 5, 10, 20, 50]
    k_range = [20]
    model_options = ["GCN", "GAT", "GIN", "SAGE"]
    # model_options = ["GIN"]

    # Load the hyperparameters from the file
    hp_path = join("results", "hp_results_pregnancy.json")
    if not os.path.exists(hp_path):
        with open(hp_path, "w") as f:
            json.dump({}, f, indent=4)
    with open(hp_path, "r") as f:
        hp_tried = json.load(f)

    # Create optuna study for hyperparameter optimization
    study = optuna.create_study(direction="maximize")

    # Load datasets in advance
    datasets = {}
    for k in k_range:
        # Load the raw data
        edge_index, color_indices, labels, split = load_pregnancy_data(k=k)

        # Find n_colors by the length of the labels
        n_colors = len(labels)

        # Set the test colors list
        test_nodes = split == 1
        test_colors = color_indices[test_nodes].unique()

        # Save to the dictionary
        datasets[k] = edge_index, color_indices, labels, test_colors, n_colors

    # Define the objective function for optuna
    def objective(trial):
        # Sample hyperparameters from the defined ranges
        hps = {
            "color_embedding_dim": trial.suggest_categorical("color_embedding_dim", color_embedding_dim_range),
            "gnn_embedding_dim": trial.suggest_categorical("gnn_embedding_dim", gnn_embedding_dim_range),
            "gnn_hidden_dim": trial.suggest_categorical("gnn_hidden_dim", gnn_hidden_dim_range),
            "k_gnn_layers": trial.suggest_categorical("k_gnn_layers", k_gnn_layers_range),
            "gnn_dropout_rate": trial.suggest_categorical("gnn_dropout_rate", gnn_dropout_rate_range),
            "gnn_mlp_hidden_dims": trial.suggest_categorical("gnn_mlp_hidden_dims", gnn_mlp_hidden_dims_range),
            "mlp_dropout_rate": trial.suggest_categorical("mlp_dropout_rate", mlp_dropout_rate_range),
            "weight_decay": trial.suggest_categorical("weight_decay", weight_decay_range),
            "attention_hidden_dims": trial.suggest_categorical("attention_hidden_dims", attention_hidden_dims_range),
            "attention_dropout_rate": trial.suggest_categorical("attention_dropout_rate", attention_dropout_rate_range),
            "n_layers": trial.suggest_categorical("n_layers", n_layers_range),
            "k": trial.suggest_categorical("k", k_range),
            "model": trial.suggest_categorical("model", model_options)
        }

        print(f"Trail: {trial.number}")

        # Skip configurations that were already tried
        if str(hps) in hp_tried:
            return hp_tried[str(hps)]

        # Load the dataset
        edge_index, color_indices, labels, test_colors, n_colors = datasets[hps["k"]]

        # Train the model with the current test color
        (test_gnn_auc, val_gnn_auc, train_gnn_auc,
         test_color_auc, val_color_auc, train_color_auc) = train(n_colors,
                                                                 data=(edge_index, color_indices,
                                                                       labels, test_colors), hps=hps, plot=True)
        # Select hyperparameters on validation AUC; test AUC is logged but never optimized on
        total_auc = (val_gnn_auc + val_color_auc) / 2

        # Save results row to CSV
        csv_path = join("results", "trial_results_pregnancy.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f,
                                    fieldnames=["trial", "train_gnn_auc", "val_gnn_auc", "test_gnn_auc",
                                                "train_color_auc", "val_color_auc", "test_color_auc"])
            if write_header:
                writer.writeheader()
            writer.writerow({"trial": trial.number, "train_gnn_auc": round(train_gnn_auc, 4),
                             "val_gnn_auc": round(val_gnn_auc, 4), "test_gnn_auc": round(test_gnn_auc, 4),
                             "train_color_auc": round(train_color_auc, 4), "val_color_auc": round(val_color_auc, 4),
                             "test_color_auc": round(test_color_auc, 4)})

        # Save the results to the hp_tried dictionary
        hp_tried[str(hps)] = total_auc

        print("\nMean AUC:", total_auc)

        print("*" * 80, "\n")

        # Save the updated hp_tried dictionary to the file
        with open(hp_path, "w") as f:
            json.dump(hp_tried, f, indent=4)

        return total_auc

    # Run the optimization for a specified number of trials
    study.optimize(objective, n_trials=1000)


def main():
    hyper_parameters_optimization()


if __name__ == '__main__':
    main()
