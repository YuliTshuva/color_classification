"""
Yuli Tshuva
Helper functions.
"""

# Imports
from constants import *
import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = "Times New Roman"


def generate_random_data(n_vertices, n_colors, n_classes):
    # Create edges between vertices
    edge_index = torch.randint(0, n_vertices, (2, 15 * n_vertices))  # 5000 random edges
    # Create random color indices for each vertex
    color_indices = torch.randint(0, n_colors, (n_vertices,))
    # Create random labels for each vertex
    labels = torch.randint(0, n_classes, (n_colors,))
    return edge_index, color_indices, labels


def load_twitch_data():
    # Load the Twitch dataset
    edges_df = pd.read_csv(EDGES_PATH)
    nodes_df = pd.read_csv(NODES_PATH)

    # Create edge_index from edges_df
    edge_index = torch.tensor(edges_df.values.T, dtype=torch.long)

    # Get the languages
    languages = nodes_df['language'].unique()
    language_to_index = {lang: idx for idx, lang in enumerate(languages)}
    color_indices = torch.tensor(nodes_df['language'].map(language_to_index).values, dtype=torch.long)

    # Set the labels
    labels = torch.tensor([1 if lang in LATIN_LANGUAGES else 0 for lang in languages], dtype=torch.long)

    return edge_index, color_indices, labels


def load_vae_extracted_data():
    """
    edge_index: (2, num_edges)
    color_indices: (num_vertices,)
    labels: (num_colors,)
    :return:
    """
    # Load the edges
    with open(VAE_EDGES, "r") as f:
        edges_text = f.read()
    edges = []
    for edge in edges_text.split("\n"):
        if len(edge) == 0:
            continue
        source, target = map(int, edge.strip().split())
        edges.append((source, target))
    edge_index = torch.tensor(edges, dtype=torch.long).T  # Shape: (2, num_edges)

    # Load the color indices
    with open(VAE_NODES, "r") as f:
        color_indices_text = f.read()
    color_indices = []
    for line in color_indices_text.split("\n"):
        if len(line) == 0:
            continue
        color_indices.append(int(line.strip().split()[1]))  # Assuming the format is "node_id color_index"
    color_indices = torch.tensor(color_indices, dtype=torch.long)  # Shape: (num_vertices,)

    # Load the labels
    with open(VAE_LABELS, "r") as f:
        labels_text = f.read()
    labels = []
    for line in labels_text.split("\n"):
        if len(line) == 0:
            continue
        labels.append(int(line.strip().split()[1]))  # Assuming the format is "color_index label"
    labels = torch.tensor(labels, dtype=torch.long)  # Shape: (num_colors,)

    return edge_index, color_indices, labels



def twitch_data_analysis():
    # Load the Twitch dataset
    _, color_indices, labels = load_twitch_data()
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
    plt.show()
