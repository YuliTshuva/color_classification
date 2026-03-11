"""
Yuli Tshuva
"""

# Imports
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import roc_auc_score, roc_curve
from constants import *
import numpy as np

# Constants
RESULTS_PATH = join("results", "results_twitch_second_model.csv")
rcParams["font.family"] = "Times New Roman"


def plot_roc():
    # Load results
    results_df = pd.read_csv(RESULTS_PATH)

    # Get the scores and labels for each model
    scores = results_df["attention_score"].values
    labels = [int(a[1]) for a in results_df["test_labels"].values]

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


def plot_scores():
    # Select relevant test color
    test_id = 3

    # Load the test scores
    test_scores = np.load(join(SCORES_DIR, f"gnn_test_scores_{test_id}.npy"))

    # Plot the scores
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(test_scores)), test_scores, color="dodgerblue", s=100)
    plt.title(f"Test scores (test color = {test_id})", fontsize=20)
    plt.xlabel("Node", fontsize=15)
    plt.ylabel("Score", fontsize=15)
    plt.show()


def main():
    plot_scores()


if __name__ == "__main__":
    main()
