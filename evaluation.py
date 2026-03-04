"""
Yuli Tshuva
"""

# Imports
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import roc_auc_score, roc_curve

# Constants
RESULTS_PATH = join("results", "results.csv")
rcParams["font.family"] = "Times New Roman"

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
