# Experiment 10 - graph construction (pregnancy)

Readout frozen at exp08/exp09 mean-pooling, same-color relation untouched, backbone
hyperparameter space identical to exp09. Only the kNN relation varies. Configurations
selected on validation AUC only; test AUC is never optimized on. Test AUC is the mean ± standard deviation over 5 seeds (only model init and dropout vary; the split is fixed).

| Backbone | Graph | What was done | Best graph | Mean deg | Configs | Val AUC (node/color) | Test AUC node | Test AUC color |
|---|---|---|---|---|---|---|---|---|
| rgat | `hybrid` | 2.2 $k_{core}$NN core plus every candidate above a similarity threshold $\tau$; degree adapts to local density, and the similarity space is swept | raw, q=0.2, k_core=3, tau=0.9990, merged | 62.8 | 95 | 0.878 / 0.863 | 0.817 ± 0.006 | 0.800 ± 0.003 |
| rgat | `knn20` | Control (exp08). Cosine $k$NN at a fixed $k=20$; every node gets exactly 20 neighbours | raw, k=20 | 31.9 | 40 | 0.875 / 0.862 | 0.810 ± 0.007 | 0.796 ± 0.002 |
| rgcn | `hybrid` | 2.2 $k_{core}$NN core plus every candidate above a similarity threshold $\tau$; degree adapts to local density, and the similarity space is swept | raw, q=0.2, k_core=1, tau=0.9990, merged | 62.1 | 83 | 0.881 / 0.860 | 0.789 ± 0.003 | 0.785 ± 0.002 |
| rgcn | `knn20` | Control (exp08). Cosine $k$NN at a fixed $k=20$; every node gets exactly 20 neighbours | raw, k=20 | 31.9 | 38 | 0.872 / 0.857 | 0.776 ± 0.001 | 0.771 ± 0.002 |

## Threshold sweep

Mean and best validation score over every backbone configuration that used each
quantile - does graph density matter, independently of which config won? Broken
down within each space, since tau is not comparable across spaces.

| Backbone | Space | q | tau | Mean deg | Configs | Val (mean) | Val (best) | Test color (best) |
|---|---|---|---|---|---|---|---|---|
| rgat | `centered` | 0.20 | 0.9346 | 73.6 | 7 | 0.858 | 0.867 | 0.809 |
| rgat | `centered` | 0.40 | 0.9771 | 55.8 | 1 | 0.774 | 0.774 | 0.763 |
| rgat | `centered` | 0.70 | 0.9932 | 28.6 | 1 | 0.845 | 0.845 | 0.790 |
| rgat | `centered` | 0.80 | 0.9955 | 18.3 | 3 | 0.854 | 0.867 | 0.797 |
| rgat | `centered` | 0.90 | 0.9972 | 9.6 | 1 | 0.859 | 0.859 | 0.783 |
| rgat | `raw` | 0.20 | 0.9990 | 62.5 | 30 | 0.856 | 0.871 | 0.809 |
| rgat | `raw` | 0.30 | 0.9992 | 54.4 | 6 | 0.848 | 0.856 | 0.801 |
| rgat | `raw` | 0.40 | 0.9994 | 46.5 | 7 | 0.837 | 0.868 | 0.808 |
| rgat | `raw` | 0.50 | 0.9995 | 38.8 | 11 | 0.855 | 0.868 | 0.807 |
| rgat | `raw` | 0.60 | 0.9995 | 30.5 | 4 | 0.856 | 0.867 | 0.805 |
| rgat | `raw` | 0.70 | 0.9996 | 23.9 | 5 | 0.866 | 0.868 | 0.810 |
| rgat | `raw` | 0.80 | 0.9996 | 15.6 | 2 | 0.851 | 0.862 | 0.806 |
| rgat | `raw` | 0.90 | 0.9997 | 9.1 | 5 | 0.853 | 0.864 | 0.806 |
| rgat | `whitened` | 0.20 | 0.4472 | 51.0 | 6 | 0.850 | 0.868 | 0.798 |
| rgat | `whitened` | 0.30 | 0.4617 | 43.8 | 1 | 0.833 | 0.833 | 0.770 |
| rgat | `whitened` | 0.40 | 0.4748 | 37.5 | 1 | 0.839 | 0.839 | 0.789 |
| rgat | `whitened` | 0.60 | 0.5076 | 24.5 | 3 | 0.828 | 0.860 | 0.792 |
| rgat | `whitened` | 0.90 | 0.5994 | 6.4 | 1 | 0.827 | 0.827 | 0.782 |
| rgcn | `centered` | 0.20 | 0.9345 | 73.7 | 1 | 0.814 | 0.814 | 0.776 |
| rgcn | `centered` | 0.30 | 0.9621 | 64.9 | 6 | 0.862 | 0.867 | 0.784 |
| rgcn | `centered` | 0.40 | 0.9771 | 55.9 | 10 | 0.851 | 0.867 | 0.784 |
| rgcn | `centered` | 0.70 | 0.9932 | 28.0 | 2 | 0.828 | 0.856 | 0.774 |
| rgcn | `centered` | 0.80 | 0.9954 | 21.3 | 2 | 0.862 | 0.868 | 0.784 |
| rgcn | `centered` | 0.90 | 0.9972 | 9.6 | 1 | 0.853 | 0.853 | 0.765 |
| rgcn | `raw` | 0.20 | 0.9990 | 62.2 | 5 | 0.844 | 0.871 | 0.786 |
| rgcn | `raw` | 0.30 | 0.9992 | 54.2 | 9 | 0.851 | 0.870 | 0.789 |
| rgcn | `raw` | 0.40 | 0.9994 | 45.4 | 2 | 0.831 | 0.834 | 0.782 |
| rgcn | `raw` | 0.50 | 0.9995 | 37.9 | 9 | 0.852 | 0.870 | 0.787 |
| rgcn | `raw` | 0.60 | 0.9995 | 30.8 | 4 | 0.859 | 0.863 | 0.777 |
| rgcn | `raw` | 0.70 | 0.9996 | 23.3 | 2 | 0.844 | 0.856 | 0.765 |
| rgcn | `raw` | 0.80 | 0.9996 | 15.6 | 2 | 0.838 | 0.842 | 0.766 |
| rgcn | `raw` | 0.90 | 0.9997 | 9.6 | 2 | 0.864 | 0.868 | 0.785 |
| rgcn | `whitened` | 0.20 | 0.4471 | 51.1 | 2 | 0.861 | 0.866 | 0.790 |
| rgcn | `whitened` | 0.30 | 0.4611 | 44.1 | 3 | 0.856 | 0.866 | 0.793 |
| rgcn | `whitened` | 0.40 | 0.4749 | 37.5 | 11 | 0.853 | 0.869 | 0.793 |
| rgcn | `whitened` | 0.50 | 0.4909 | 30.6 | 4 | 0.857 | 0.865 | 0.800 |
| rgcn | `whitened` | 0.60 | 0.5063 | 25.1 | 1 | 0.864 | 0.864 | 0.796 |
| rgcn | `whitened` | 0.80 | 0.5559 | 12.5 | 3 | 0.857 | 0.861 | 0.798 |
| rgcn | `whitened` | 0.90 | 0.5902 | 9.0 | 2 | 0.862 | 0.862 | 0.785 |

## Similarity space

The MiniLM embeddings are anisotropic (mean cosine between two random pregnancy
nodes: +0.979), so `raw` neighbourhoods are decided in the fourth decimal place.
`centered` and `whitened` strip that common component.

| Backbone | Space | Mean deg | Configs | Val (mean) | Val (best) | Test color (best) |
|---|---|---|---|---|---|---|
| rgat | `centered` | 51.1 | 13 | 0.850 | 0.867 | 0.809 |
| rgat | `raw` | 46.8 | 70 | 0.854 | 0.871 | 0.810 |
| rgat | `whitened` | 38.9 | 12 | 0.840 | 0.868 | 0.798 |
| rgcn | `centered` | 51.4 | 22 | 0.851 | 0.868 | 0.784 |
| rgcn | `raw` | 41.5 | 35 | 0.849 | 0.871 | 0.789 |
| rgcn | `whitened` | 32.7 | 26 | 0.856 | 0.869 | 0.800 |

## Best configuration per graph

- **rgat/hybrid**: `{"gnn_embedding_dim": 156, "gnn_hidden_dim": 128, "k_gnn_layers": 3, "gnn_dropout_rate": 0.0, "gnn_mlp_hidden_dims": 50, "mlp_dropout_rate": 0.2, "color_mlp_hidden_dims": 50, "weight_decay": 0.001, "feature_norm": "standardize", "graph": "hybrid", "backbone": "rgat", "num_heads": 2, "sim_quantile": 0.2, "k_core": 3, "split_core_relation": false, "sim_space": "raw"}`
- **rgat/knn20**: `{"gnn_embedding_dim": 128, "gnn_hidden_dim": 100, "k_gnn_layers": 2, "gnn_dropout_rate": 0.1, "gnn_mlp_hidden_dims": 64, "mlp_dropout_rate": 0.1, "color_mlp_hidden_dims": 32, "weight_decay": 0.0, "feature_norm": "standardize", "graph": "knn20", "backbone": "rgat", "num_heads": 4, "k": 20}`
- **rgcn/hybrid**: `{"gnn_embedding_dim": 156, "gnn_hidden_dim": 100, "k_gnn_layers": 3, "gnn_dropout_rate": 0.1, "gnn_mlp_hidden_dims": 50, "mlp_dropout_rate": 0.0, "color_mlp_hidden_dims": 32, "weight_decay": 0.001, "feature_norm": "standardize", "graph": "hybrid", "backbone": "rgcn", "sim_quantile": 0.2, "k_core": 1, "split_core_relation": false, "sim_space": "raw"}`
- **rgcn/knn20**: `{"gnn_embedding_dim": 156, "gnn_hidden_dim": 156, "k_gnn_layers": 2, "gnn_dropout_rate": 0.0, "gnn_mlp_hidden_dims": 64, "mlp_dropout_rate": 0.2, "color_mlp_hidden_dims": 64, "weight_decay": 0.0, "feature_norm": "standardize", "graph": "knn20", "backbone": "rgcn", "k": 20}`
