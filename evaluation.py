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
RESULTS_PATH = join("results", "results_twitch_third_model.csv")
rcParams["font.family"] = "Times New Roman"
NUM_COLORS = 21


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


def main():
    find_optimal_threshold()


if __name__ == "__main__":
    main()
