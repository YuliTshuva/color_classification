from train06 import DISEASES
from utils import load_minilm, load_disease_data

# Set a list for graphs storage
graphs = []

# Load miniLM
edge_index, color_indices, labels, split = load_minilm()
graphs.append((edge_index, color_indices, labels, split))

# Load the diseases
for disease in DISEASES:
    edge_index, color_indices, labels, split = load_disease_data(disease=disease)
    graphs.append((edge_index, color_indices, labels, split))

# Please analyze each of the loaded graph as follows:
#   Print graph index in the list
#   Print amount of edges
#   Print average node degree
#   Print amount of nodes
#   Print amount of colors
#   Print mean nodes per color (sanity check)

# Cluade, please implement from here:

for i, (edge_index, color_indices, labels, split) in enumerate(graphs):
    num_edges = edge_index.shape[1]
    num_nodes = color_indices.shape[0]
    num_colors = labels.shape[0]
    avg_degree = num_edges / num_nodes
    mean_nodes_per_color = num_nodes / num_colors

    print(f"Graph {i}:")
    print(f"  Edges:               {num_edges}")
    print(f"  Average node degree: {avg_degree:.2f}")
    print(f"  Nodes:               {num_nodes}")
    print(f"  Colors:              {num_colors}")
    print(f"  Mean nodes per color:{mean_nodes_per_color:.2f}")
    print()
