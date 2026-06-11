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
import optuna
from tqdm.auto import tqdm
import os

# Create a random seed for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set the font for the plots
rcParams['font.family'] = "Times New Roman"

# Hyperparameters
EPOCHS = 10000
LR = 0.001
TOLERANCE = 30
DISEASES = ['acute_and_unspecified_renal_failure', 'cardiac_dysrhythmias', 'congestive_heart_failure_nonhypertensive',
            'coronary_atherosclerosis_and_other_heart_disease', 'disorders_of_lipid_metabolism',
            'essential_hypertension', 'fluid_and_electrolyte_disorders',
            'hypertension_with_complications_and_secondary_hypertension']


def train(n_colors, data, hps, directed=False):
    """
    Get the data and train the models for color classification.
    :param data: edge_index (2, edges), color_indices (n_vertices,), labels (n_colors,)
    :param n_colors: Amount of colors in the dataset.
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
    alpha = hps["alpha"]
    model = hps["model"]

    # Initiate the GNN model
    if model == "RGCN":
        gnn_model = RGCN(in_dim=color_embedding_dim,
                         hidden_dim=gnn_hidden_dim,
                         out_dim=gnn_embedding_dim,
                         num_relations=2,
                         num_layers=k_gnn_layers,
                         dropout=gnn_dropout_rate,
                         directed=directed)
    elif model == "RGAT":
        gnn_model = RGAT(in_dim=color_embedding_dim,
                         hidden_dim=gnn_hidden_dim,
                         out_dim=gnn_embedding_dim,
                         num_relations=2,
                         num_layers=k_gnn_layers,
                         dropout=gnn_dropout_rate,
                         directed=directed)
    else:
        raise ValueError(f"Invalid model type: {model}. Choose 'RGCN' or 'RGAT'.")

    # Initiate the MLP model for GNN prediction
    mlp_model = MLP(input_dim=gnn_embedding_dim,
                    hidden_dims=gnn_mlp_hidden_dims,
                    output_dim=1,
                    dropout_rate=mlp_dropout_rate)

    # Initiate the color embedding model
    color_embedding_model = ColorEmbedding(n_colors=n_colors,
                                           embedding_dim=color_embedding_dim)

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
    for epoch in tqdm(range(EPOCHS)):
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
        loss = alpha * loss_gnn + total_variance  # Maximize over the variance of the colors

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
    # gnn_predictions = (gnn_scores > 0.5).long()

    # Calculate the accuracies for the attention model and the gnn
    node_labels = node_labels.long()
    # train_gnn_accuracy = (gnn_predictions[training_mask] == node_labels[training_mask]).float().mean().item()
    # test_gnn_accuracy = (gnn_predictions[~training_mask] == node_labels[~training_mask]).float().mean().item()

    # Calculate AUC-ROC for the attention model and the gnn over the training set
    train_gnn_auc = roc_auc_score(node_labels[training_mask].cpu().numpy(), gnn_scores[training_mask].cpu().numpy())
    test_gnn_auc = roc_auc_score(node_labels[~training_mask].cpu().numpy(), gnn_scores[~training_mask].cpu().numpy())

    return test_gnn_auc, train_gnn_auc


def analyze_results():
    # Load the hyperparameters from the file
    with open(join("results", "hp_results.json"), "r") as f:
        hp_tried = json.load(f)

    # Find the best hyperparameters based on the sum of test_gnn_auc and test_gnn_accuracy
    best_hps = None
    best_score = -float('inf')
    for hps_str, score in hp_tried.items():
        if score > best_score:
            best_score = score
            best_hps = hps_str
    print(f"Best Hyperparameters: {best_hps}")
    print("Best average AUC:", best_score)
    return best_hps


def hyper_parameters_optimization():
    # Set the hyperparameter ranges for the grid search
    color_embedding_dim_range = [3, 5, 8, 16]
    gnn_embedding_dim_range = [16, 32, 64]
    gnn_hidden_dim_range = [16, 32, 64]
    k_gnn_layers_range = [3, 8, 15, 25]
    gnn_dropout_rate_range = [0.1, 0.3, 0.5, 0.7]
    gnn_mlp_hidden_dims_range = [4, 8, 16, 32]
    mlp_dropout_rate_range = [0.0, 0.1, 0.2]
    alpha_range = [1000]
    k_range = [3, 5, 10, 20, 50]
    model_options = ["RGCN", "RGAT"]

    # Load the hyperparameters from the file
    hp_path = join("results", "hp_results.json")
    if not os.path.exists(hp_path):
        with open(hp_path, "w") as f:
            json.dump({}, f, indent=4)
    with open(hp_path, "r") as f:
        hp_tried = json.load(f)

    # Create optuna study for hyperparameter optimization
    study = optuna.create_study(direction="maximize")

    # Load datasets in advance
    datasets = {}
    for disease in DISEASES:
        datasets[disease] = {}
        for k in k_range:
            # Load the raw data
            edge_index, color_indices, labels, split = load_disease_data(disease=disease, k=k)

            # Find n_colors by the length of the labels
            n_colors = len(labels)

            # Set the test colors list
            test_nodes = split == 1
            test_colors = color_indices[test_nodes].unique()

            # Add edges between nodes of the same color
            same_color_edges = []
            for color in range(n_colors):
                # Find all nodes with the current color index
                color_nodes = (color_indices == color).nonzero(as_tuple=True)[0]
                if len(color_nodes) > 1:
                    # Create edges between all pairs of nodes with the same color
                    for i in range(len(color_nodes)):
                        for j in range(i + 1, len(color_nodes)):
                            same_color_edges.append((color_nodes[i].item(), color_nodes[j].item()))
                else:
                    raise Exception(f"color {color} has one or less nodes.")

            # Find the lengths of both edge_index and same_color_edges
            r1, r2 = edge_index.size(1), len(same_color_edges)

            # Convert the same color edges to a tensor and concatenate with the original edge_index
            if same_color_edges:
                same_color_edge_index = torch.tensor(same_color_edges, dtype=torch.long).t().contiguous()
                edge_index = torch.cat([edge_index, same_color_edge_index], dim=1)
            else:
                raise Exception("No same color edges were created. Check the dataset and the color indices.")

            # Create a tensor that indicates which relation each edge belongs to (0 for original edges, 1 for same color edges)
            edge_relation = torch.cat([torch.zeros(r1, dtype=torch.long), torch.ones(r2, dtype=torch.long)], dim=0)

            # Save to the dictionary
            datasets[disease][k] = edge_index, edge_relation, color_indices, labels, test_colors, n_colors

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
            "alpha": trial.suggest_categorical("alpha", alpha_range),
            "k": trial.suggest_categorical("k", k_range),
            "model": trial.suggest_categorical("model", model_options)
        }

        # Store the results for each disease
        total_auc = 0

        print(f"Trail: {trial.number}")

        for disease in DISEASES:
            # Load the dataset
            edge_index, edge_relation, color_indices, labels, test_colors, n_colors = datasets[disease][hps["k"]]

            if str(hps) in hp_tried:
                return hp_tried[str(hps)]

            # Train the model with the current test color
            test_gnn_auc, train_gnn_auc = train(n_colors, data=(edge_index, edge_relation, color_indices,
                                                 labels, test_colors), hps=hps)
            total_auc += test_gnn_auc

            # Print
            print(f"\tDisease: {disease}\tTest GNN AUC: {test_gnn_auc:.4f}")

        # Normalize the total AUC by the number of diseases tested
        total_auc /= len(DISEASES)

        # Save the results to the hp_tried dictionary
        hp_tried[str(hps)] = total_auc

        print("\nMean AUC:", total_auc)

        print("*"*80, "\n")

        # Save the updated hp_tried dictionary to the file
        with open(join("results", "hp_results.json"), "w") as f:
            json.dump(hp_tried, f, indent=4)

        return total_auc

    # Run the optimization for a specified number of trials
    study.optimize(objective, n_trials=1000)


def main():
    hyper_parameters_optimization()


if __name__ == '__main__':
    main()
