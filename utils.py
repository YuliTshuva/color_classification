"""
Yuli Tshuva
Helper functions.
"""

# Imports
from constants import *
import torch
import pandas as pd


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
