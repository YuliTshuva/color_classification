"""
Yuli Tshuva
"""

# Imports
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import json
from matplotlib import rcParams
from models import *
from sklearn.decomposition import PCA

rcParams["font.family"] = "Times New Roman"

# Constants
RESULTS_PATH = join("results", "results_twitch_model_6.csv")
rcParams["font.family"] = "Times New Roman"
NUM_COLORS = 21


def plot_roc():
    # Load results
    results_df = pd.read_csv(RESULTS_PATH)

    # Get the scores and labels for each model
    labels, scores = results_df["test_label"].values, results_df["test_gnn_accuracy"].values
    new_scores = []
    for score, label in zip(scores, labels):
        if label == 1:
            new_scores.append(score)
        else:
            new_scores.append(1 - score)
    scores = np.array(new_scores)

    # Calculate AUC-ROC for each model
    auc_roc = roc_auc_score(labels, scores)
    # Plot the ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    # Print the AUC-ROC score
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", label=f"AUC-ROC = {auc_roc:.4f}")
    plt.plot([0, 1], [0, 1], color="red", linestyle="--")
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.title("ROC Curve", fontsize=20)
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


def find_optimal_threshold():
    # Load the results df
    results_df = pd.read_csv(RESULTS_PATH)
    # Get the test labels
    labels = [int(a[1]) for a in results_df["test_labels"].values]
    # Set a dictionary for the scores
    scores_arrays = []
    for test_id in range(NUM_COLORS):
        # Load the test scores
        scores_arrays.append(np.load(join(SCORES_DIR, f"gnn_test_scores_{test_id}.npy")))
    # Define the default threshold
    default_threshold = 0.5
    # Set the step size
    step_size = 0.01
    # Set a list to store the accuracies for each threshold
    best_acc, best_thresh = 0, default_threshold
    # Iterate over the thresholds
    for thresh in np.arange(default_threshold, 1 + step_size, step_size):
        # Calculate the predictions based on the current threshold
        predictions = [scores > thresh for scores in scores_arrays]
        # Calculate the accuracy for the current threshold
        correct_predictions = [np.mean(predictions[i] == labels[i]) > 0.5 for i in range(NUM_COLORS)]
        # Calculate the average accuracy across all colors
        avg_accuracy = np.sum(correct_predictions)
        # Update the best accuracy and threshold if the current average accuracy is better
        if avg_accuracy > best_acc:
            best_acc = avg_accuracy
            best_thresh = thresh
    print(f"Optimal Threshold: {best_thresh:.2f}, Best Accuracy: {best_acc}/{NUM_COLORS} ({best_acc / NUM_COLORS:.4f})")


def sort_hp_results():
    # Load the json file
    with open(join("results", "hp_results.json"), "r") as f:
        hp_results = json.load(f)
    # Sort the results by accuracy
    sorted_results = sorted(hp_results.items(), key=lambda x: x[1]["sum_acc"], reverse=True)
    # Construct a df from the sorted results
    sorted_df = pd.DataFrame(columns=list(hp_results["1"]["config"].keys()) + ["sum_acc"] + ["rank"])
    for config, result in sorted_results:
        row = list(result["config"].values()) + [result["sum_acc"]] + [config]
        sorted_df.loc[len(sorted_df)] = row
    # Save the sorted results to a csv file
    sorted_df.to_csv(join("results", "sorted_hp_results.csv"), index=True)


def main():
    sort_hp_results()


def manual_auc():
    df = pd.read_csv(join("results", "results_twitch_fifth_model.csv"))
    auc = roc_auc_score(df["test_labels"], df["scores"])
    # Plot the ROC curve
    fpr, tpr, thresholds = roc_curve([int(a[1]) for a in df["test_labels"]], df["scores"].astype(float))
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", label=f"AUC-ROC")
    plt.plot([0, 1], [0, 1], color="red", linestyle="--")
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.title(f"ROC Curve (AUC: {auc:.4f})", fontsize=20)
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


def plot_learnt_colors_embeddings():
    # Initiate a color embeddings model
    n_colors = 21
    neg_colors = [2, 3, 4, 10, 12, 15]
    pos_colors = [0, 1, 5, 6, 7, 8, 9, 11, 13, 14, 16, 17, 18, 19, 20]
    color_embedding_model = ColorEmbedding(n_colors=n_colors,
                                           embedding_dim=COLOR_EMBEDDING_DIM)

    embeddings = []
    for train_id in range(2):
        # Load the model state dict
        model_path = join(MODELS_DIR, f"color_embedding_model_{train_id}.pth")
        color_embedding_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        # Get the embeddings
        with torch.no_grad():
            embedding = color_embedding_model(torch.tensor(range(n_colors))).cpu().numpy()
            embeddings.append(embedding)
    embeddings = np.mean(embeddings, axis=0)

    # Plot the embeddings in 2D using PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[neg_colors, 0], embeddings_2d[neg_colors, 1], color="red", label="Negative Colors", s=100)
    plt.scatter(embeddings_2d[pos_colors, 0], embeddings_2d[pos_colors, 1], color="blue", label="Positive Colors",
                s=100)
    plt.xlabel("Principal Component 1", fontsize=15)
    plt.ylabel("Principal Component 2", fontsize=15)
    plt.title("Learned Color Embeddings (PCA)", fontsize=20)
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    manual_auc()
    plot_learnt_colors_embeddings()
