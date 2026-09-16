# Experiment 10 - graph construction (pregnancy_C)

Readout frozen at exp08/exp09 mean-pooling, same-color relation untouched, backbone
hyperparameter space identical to exp09. Only the kNN relation varies. Configurations
selected on validation AUC only; test AUC is never optimized on. Test AUC is the mean ± standard deviation over 5 seeds (only model init and dropout vary; the split is fixed).

| Backbone | Graph | What was done | Best graph | Mean deg | Configs | Val AUC (node/color) | Test AUC node | Test AUC color |
|---|---|---|---|---|---|---|---|---|
| rgat | `hybrid` | 2.2 $k_{core}$NN core plus every candidate above a similarity threshold $\tau$; degree adapts to local density, and the similarity space is swept | whitened, q=0.8, k_core=1, tau=0.5866, merged | 11.7 | 93 | 0.835 / 0.834 | 0.787 ± 0.008 | 0.780 ± 0.005 |
| rgat | `knn20` | Control (exp08). Cosine $k$NN at a fixed $k=20$; every node gets exactly 20 neighbours | raw, k=20 | 27.2 | 38 | 0.799 / 0.823 | 0.767 ± 0.008 | 0.767 ± 0.004 |

## Threshold sweep

Mean and best validation score over every backbone configuration that used each
quantile - does graph density matter, independently of which config won? Broken
down within each space, since tau is not comparable across spaces.

| Backbone | Space | q | tau | Mean deg | Configs | Val (mean) | Val (best) | Test color (best) |
|---|---|---|---|---|---|---|---|---|
| rgat | `centered` | 0.20 | 0.8949 | 48.6 | 1 | 0.761 | 0.761 | 0.759 |
| rgat | `centered` | 0.40 | 0.9129 | 35.3 | 4 | 0.782 | 0.804 | 0.770 |
| rgat | `centered` | 0.50 | 0.9210 | 28.6 | 1 | 0.779 | 0.779 | 0.756 |
| rgat | `centered` | 0.70 | 0.9361 | 16.9 | 2 | 0.796 | 0.804 | 0.768 |
| rgat | `centered` | 0.80 | 0.9436 | 12.2 | 5 | 0.790 | 0.807 | 0.769 |
| rgat | `centered` | 0.90 | 0.9552 | 6.2 | 1 | 0.692 | 0.692 | 0.755 |
| rgat | `raw` | 0.20 | 0.9617 | 50.9 | 2 | 0.791 | 0.792 | 0.765 |
| rgat | `raw` | 0.30 | 0.9667 | 43.9 | 2 | 0.786 | 0.789 | 0.780 |
| rgat | `raw` | 0.40 | 0.9705 | 37.0 | 1 | 0.800 | 0.800 | 0.766 |
| rgat | `raw` | 0.80 | 0.9819 | 11.3 | 4 | 0.813 | 0.819 | 0.770 |
| rgat | `raw` | 0.90 | 0.9852 | 6.2 | 2 | 0.781 | 0.797 | 0.759 |
| rgat | `whitened` | 0.20 | 0.4685 | 51.0 | 13 | 0.804 | 0.823 | 0.794 |
| rgat | `whitened` | 0.30 | 0.4856 | 43.5 | 4 | 0.816 | 0.818 | 0.793 |
| rgat | `whitened` | 0.40 | 0.5022 | 36.5 | 5 | 0.798 | 0.816 | 0.780 |
| rgat | `whitened` | 0.50 | 0.5192 | 29.8 | 5 | 0.806 | 0.820 | 0.787 |
| rgat | `whitened` | 0.60 | 0.5363 | 24.1 | 8 | 0.784 | 0.819 | 0.786 |
| rgat | `whitened` | 0.70 | 0.5590 | 17.5 | 5 | 0.807 | 0.822 | 0.779 |
| rgat | `whitened` | 0.80 | 0.5857 | 11.9 | 24 | 0.810 | 0.834 | 0.783 |
| rgat | `whitened` | 0.90 | 0.6221 | 7.6 | 4 | 0.810 | 0.821 | 0.782 |

## Similarity space

The MiniLM embeddings are anisotropic (mean cosine between two random pregnancy
nodes: +0.979), so `raw` neighbourhoods are decided in the fourth decimal place.
`centered` and `whitened` strip that common component.

| Backbone | Space | Mean deg | Configs | Val (mean) | Val (best) | Test color (best) |
|---|---|---|---|---|---|---|
| rgat | `centered` | 22.8 | 14 | 0.779 | 0.807 | 0.770 |
| rgat | `raw` | 25.8 | 11 | 0.797 | 0.819 | 0.780 |
| rgat | `whitened` | 26.0 | 68 | 0.805 | 0.834 | 0.794 |

## Best configuration per graph

- **rgat/hybrid**: `{"gnn_embedding_dim": 100, "gnn_hidden_dim": 156, "k_gnn_layers": 3, "gnn_dropout_rate": 0.0, "gnn_mlp_hidden_dims": 80, "mlp_dropout_rate": 0.2, "color_mlp_hidden_dims": 32, "weight_decay": 0.001, "feature_norm": "none", "graph": "hybrid", "backbone": "rgat", "num_heads": 4, "sim_quantile": 0.8, "k_core": 1, "split_core_relation": false, "sim_space": "whitened"}`
- **rgat/knn20**: `{"gnn_embedding_dim": 100, "gnn_hidden_dim": 156, "k_gnn_layers": 3, "gnn_dropout_rate": 0.0, "gnn_mlp_hidden_dims": 50, "mlp_dropout_rate": 0.0, "color_mlp_hidden_dims": 64, "weight_decay": 0.01, "feature_norm": "none", "graph": "knn20", "backbone": "rgat", "num_heads": 4, "k": 20}`
