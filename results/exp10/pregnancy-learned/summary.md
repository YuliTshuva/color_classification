# Experiment 10 - graph construction (pregnancy-learned)

Readout frozen at exp08/exp09 mean-pooling, same-color relation untouched, backbone
hyperparameter space identical to exp09. Only the kNN relation varies. Configurations
selected on validation AUC only; test AUC is never optimized on. Test AUC is the mean ± standard deviation over 5 seeds (only model init and dropout vary; the split is fixed).

| Backbone | Graph | What was done | Best graph | Mean deg | Configs | Val AUC (node/color) | Test AUC node | Test AUC color |
|---|---|---|---|---|---|---|---|---|
| rgat | `hybrid` | 2.2 $k_{core}$NN core plus every candidate above a similarity threshold $\tau$; degree adapts to local density, and the similarity space is swept | raw, q=0.3, k_core=3, tau=0.9992, split | 54.7 | 99 | 0.780 / 0.812 | 0.683 ± 0.012 | 0.705 ± 0.020 |
| rgat | `knn20` | Control (exp08). Cosine $k$NN at a fixed $k=20$; every node gets exactly 20 neighbours | raw, k=20 | 31.9 | 47 | 0.778 / 0.762 | 0.678 ± 0.040 | 0.670 ± 0.029 |

## Threshold sweep

Mean and best validation score over every backbone configuration that used each
quantile - does graph density matter, independently of which config won? Broken
down within each space, since tau is not comparable across spaces.

| Backbone | Space | q | tau | Mean deg | Configs | Val (mean) | Val (best) | Test color (best) |
|---|---|---|---|---|---|---|---|---|
| rgat | `centered` | 0.20 | 0.9342 | 74.0 | 2 | 0.753 | 0.754 | 0.701 |
| rgat | `centered` | 0.30 | 0.9621 | 64.9 | 3 | 0.675 | 0.755 | 0.722 |
| rgat | `centered` | 0.40 | 0.9771 | 55.5 | 3 | 0.631 | 0.758 | 0.669 |
| rgat | `centered` | 0.50 | 0.9852 | 45.7 | 1 | 0.555 | 0.555 | 0.557 |
| rgat | `centered` | 0.60 | 0.9901 | 36.5 | 2 | 0.582 | 0.621 | 0.570 |
| rgat | `centered` | 0.70 | 0.9932 | 30.0 | 1 | 0.769 | 0.769 | 0.683 |
| rgat | `centered` | 0.90 | 0.9971 | 12.9 | 1 | 0.585 | 0.585 | 0.593 |
| rgat | `raw` | 0.20 | 0.9990 | 62.7 | 11 | 0.749 | 0.791 | 0.724 |
| rgat | `raw` | 0.30 | 0.9992 | 54.5 | 32 | 0.747 | 0.796 | 0.754 |
| rgat | `raw` | 0.40 | 0.9994 | 46.7 | 4 | 0.760 | 0.780 | 0.713 |
| rgat | `raw` | 0.50 | 0.9995 | 38.7 | 4 | 0.755 | 0.773 | 0.718 |
| rgat | `raw` | 0.60 | 0.9995 | 31.1 | 7 | 0.746 | 0.782 | 0.740 |
| rgat | `raw` | 0.70 | 0.9996 | 23.0 | 4 | 0.758 | 0.768 | 0.727 |
| rgat | `raw` | 0.80 | 0.9996 | 15.6 | 5 | 0.715 | 0.747 | 0.734 |
| rgat | `raw` | 0.90 | 0.9997 | 9.5 | 6 | 0.712 | 0.759 | 0.727 |
| rgat | `whitened` | 0.20 | 0.4464 | 51.4 | 3 | 0.677 | 0.753 | 0.707 |
| rgat | `whitened` | 0.30 | 0.4612 | 44.0 | 4 | 0.706 | 0.737 | 0.693 |
| rgat | `whitened` | 0.50 | 0.4902 | 30.9 | 3 | 0.618 | 0.666 | 0.640 |
| rgat | `whitened` | 0.70 | 0.5259 | 19.3 | 1 | 0.639 | 0.639 | 0.619 |
| rgat | `whitened` | 0.80 | 0.5513 | 13.9 | 2 | 0.650 | 0.669 | 0.704 |

## Similarity space

The MiniLM embeddings are anisotropic (mean cosine between two random pregnancy
nodes: +0.979), so `raw` neighbourhoods are decided in the fourth decimal place.
`centered` and `whitened` strip that common component.

| Backbone | Space | Mean deg | Configs | Val (mean) | Val (best) | Test color (best) |
|---|---|---|---|---|---|---|
| rgat | `centered` | 51.6 | 13 | 0.654 | 0.769 | 0.722 |
| rgat | `raw` | 44.1 | 73 | 0.744 | 0.796 | 0.754 |
| rgat | `whitened` | 36.2 | 13 | 0.665 | 0.753 | 0.707 |

## Best configuration per graph

- **rgat/hybrid**: `{"gnn_embedding_dim": 100, "gnn_hidden_dim": 128, "k_gnn_layers": 3, "gnn_dropout_rate": 0.2, "gnn_mlp_hidden_dims": 80, "mlp_dropout_rate": 0.2, "color_mlp_hidden_dims": 64, "weight_decay": 0.0001, "graph": "hybrid", "backbone": "rgat", "node_input": "learned", "node_input_dim": 128, "num_heads": 4, "sim_quantile": 0.3, "k_core": 3, "split_core_relation": true, "sim_space": "raw"}`
- **rgat/knn20**: `{"gnn_embedding_dim": 100, "gnn_hidden_dim": 128, "k_gnn_layers": 3, "gnn_dropout_rate": 0.1, "gnn_mlp_hidden_dims": 50, "mlp_dropout_rate": 0.1, "color_mlp_hidden_dims": 50, "weight_decay": 0.01, "graph": "knn20", "backbone": "rgat", "node_input": "learned", "node_input_dim": 128, "num_heads": 2, "k": 20}`
