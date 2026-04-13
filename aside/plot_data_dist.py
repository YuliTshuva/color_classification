"""
Yuli Tshuva
Plotting the distribution of the data for each color.
"""

# Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import *

# Constants
PLOTS_DIR = 'plots'


def main():
    # Load the Twitch dataset
    edge_index, color_indices, labels = load_twitch_data()

    ## Analyze the distribution of colors in the Twitch dataset
    # Find the number of vertices for each color
    color_counts = torch.bincount(color_indices)
    # Plot the distribution of colors
    plt.figure(figsize=(10, 6))
    # Create a bar plot for the distribution of colors in log scale
    plt.bar(range(len(color_counts)), color_counts.numpy(), color='skyblue', edgecolor='black', log=True)
    plt.xlabel('Color Index (Language)', fontsize=15)
    plt.ylabel('Number of Vertices', fontsize=15)
    plt.title('Distribution of Colors (Languages) in Twitch Dataset', fontsize=20)
    plt.xticks(range(len(color_counts)), [f'Lang {i}' for i in range(len(color_counts))], rotation=45)
    plt.tight_layout()
    plt.savefig(join(PLOTS_DIR, 'color_distribution.png'))
    plt.close()

    # Convert each element in color_indices to its corresponding label
    color_labels = labels[color_indices]
    # Find the number of vertices for each color label
    label_counts = torch.bincount(color_labels)
    # Plot the distribution of color labels
    plt.figure(figsize=(10, 6))
    # Create a bar plot for the distribution of color labels in log scale
    plt.bar(range(len(label_counts)), label_counts.numpy(), color='salmon', edgecolor='black')
    plt.xlabel('Color Label (Latin vs Non-Latin)', fontsize=15)
    plt.ylabel('Number of Vertices', fontsize=15)
    plt.title('Distribution of Color Labels (Latin vs Non-Latin) in Twitch Dataset', fontsize=20)
    plt.xticks(range(len(label_counts)), ['Non-Latin', 'Latin'])
    plt.yticks(label_counts.numpy())
    plt.tight_layout()
    plt.savefig(join(PLOTS_DIR, 'color_labels.png'))
    plt.close()

    # Convert each element in edge_index to its corresponding color label
    edge_labels = torch.cat([color_labels[edge_index[0]].unsqueeze(0), color_labels[edge_index[1]].unsqueeze(0)], dim=0)
    match_labels = torch.sum(edge_labels, dim=0)
    # Find the number of edges for each match label
    match_counts = torch.bincount(match_labels)
    # Plot the distribution of match labels
    plt.figure(figsize=(10, 6))
    # Create a bar plot for the distribution of match labels in log scale
    plt.bar(range(len(match_counts)), match_counts.numpy(), color='lightgreen', edgecolor='black')
    plt.xlabel('Match Label (0: Non-Latin to Non-Latin, 1: Latin to Non-Latin, 2: Latin to Latin)', fontsize=15)
    plt.ylabel('Number of Edges', fontsize=15)
    plt.title('Distribution of Match Labels in Twitch Dataset', fontsize=20)
    plt.xticks(range(len(match_counts)), [f'Non-Latin to Non-Latin\n{match_counts.numpy()[0]}',
                                          f'Latin to Non-Latin\n{match_counts.numpy()[1]}',
                                          f'Latin to Latin\n{match_counts.numpy()[2]}'])
    plt.tight_layout()
    plt.savefig(join(PLOTS_DIR, 'types_of_edges.png'))
    plt.close()

    # Count the number of edges between the color 15 to nodes with label 0 and label 1
    for test_color in [2, 3, 4, 10, 12, 15]:
        color_15_edge_labels = color_labels[edge_index[:, (color_indices[edge_index[0]] == test_color) | (color_indices[edge_index[1]] == test_color)]]
        color_15_match_labels = torch.sum(color_15_edge_labels, dim=0)
        color_15_match_counts = torch.bincount(color_15_match_labels)
        # Plot the distribution of match labels for color 15
        plt.figure(figsize=(10, 6))
        # Create a bar plot for the distribution of match labels for color 15 in log scale
        plt.bar(range(len(color_15_match_counts)), color_15_match_counts.numpy(), color='orchid', edgecolor='black')
        plt.xlabel('Match Label (0: Non-Latin to Non-Latin, 1: Latin to Non-Latin)', fontsize=15)
        plt.ylabel('Number of Edges', fontsize=15)
        plt.title(f'Distribution of Match Labels for Color {test_color} in Twitch Dataset', fontsize=20)
        plt.xticks(range(len(color_15_match_counts)), [f'Non-Latin to Non-Latin\n{color_15_match_counts.numpy()[0]}',
                                                      f'Latin to Non-Latin\n{color_15_match_counts.numpy()[1]}'])
        plt.tight_layout()
        plt.savefig(join(PLOTS_DIR, f'color_{test_color}_edge_types.png'))
        plt.show()


if __name__ == "__main__":
    main()
