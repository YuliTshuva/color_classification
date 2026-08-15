# Experiment 09 - learned pooling (pregnancy)

Backbone frozen at the exp08 winner (two-relation graph, MiniLM features, no color
embedding). Configurations selected on validation AUC only; test AUC is never
optimized on. Test AUC is the mean ± standard deviation over 5 seeds (only model init and dropout vary; the split is fixed).

| Backbone | Readout | What was done | Configs | Val AUC (node/color) | Test AUC node | Test AUC color |
|---|---|---|---|---|---|---|
| rgat | `mean` | Control (exp08). Parameter-free mean of the member node embeddings, then a color MLP | 42 | 0.874 / 0.866 | 0.818 ± 0.004 | 0.798 ± 0.004 |
| rgat | `gated_attn` | 1.1 Gated attention (MIL) pooling; scores softmax-normalized within each color | 46 | 0.866 / 0.865 | 0.809 ± 0.006 | 0.796 ± 0.005 |
| rgat | `node_score_mean` | Baseline. Color score is the mean of the node head's sigmoid scores; no color head | 34 | 0.872 / 0.863 | 0.810 ± 0.006 | 0.796 ± 0.002 |
| rgat | `pma` | 1.4 Pooling by Multihead Attention, optionally after self-attention among the members | 44 | 0.874 / 0.863 | 0.811 ± 0.005 | 0.795 ± 0.004 |
| rgat | `supernode` | 1.2 One virtual node per color as a third relation; its embedding feeds the color head | 42 | 0.888 / 0.865 | 0.810 ± 0.006 | 0.795 ± 0.009 |
| rgat | `multistat` | 1.3 Multi-statistic readout: mean || max || sum concatenated before the color MLP | 38 | 0.870 / 0.870 | 0.794 ± 0.007 | 0.792 ± 0.007 |

## Best configuration per readout

- **rgat/mean**: `{"gnn_embedding_dim": 128, "gnn_hidden_dim": 128, "k_gnn_layers": 3, "gnn_dropout_rate": 0.1, "gnn_mlp_hidden_dims": 50, "mlp_dropout_rate": 0.2, "weight_decay": 0.0001, "feature_norm": "standardize", "k": 20, "readout": "mean", "backbone": "rgat", "num_heads": 2, "color_mlp_hidden_dims": 32}`
- **rgat/gated_attn**: `{"gnn_embedding_dim": 156, "gnn_hidden_dim": 128, "k_gnn_layers": 2, "gnn_dropout_rate": 0.1, "gnn_mlp_hidden_dims": 64, "mlp_dropout_rate": 0.2, "weight_decay": 0.001, "feature_norm": "standardize", "k": 20, "readout": "gated_attn", "backbone": "rgat", "num_heads": 2, "color_mlp_hidden_dims": 50, "attn_pool_hidden_dim": 64, "attn_pool_dropout": 0.1}`
- **rgat/node_score_mean**: `{"gnn_embedding_dim": 100, "gnn_hidden_dim": 128, "k_gnn_layers": 3, "gnn_dropout_rate": 0.0, "gnn_mlp_hidden_dims": 80, "mlp_dropout_rate": 0.0, "weight_decay": 0.01, "feature_norm": "standardize", "k": 20, "readout": "node_score_mean", "backbone": "rgat", "num_heads": 4}`
- **rgat/pma**: `{"gnn_embedding_dim": 156, "gnn_hidden_dim": 100, "k_gnn_layers": 3, "gnn_dropout_rate": 0.2, "gnn_mlp_hidden_dims": 80, "mlp_dropout_rate": 0.1, "weight_decay": 0.01, "feature_norm": "standardize", "k": 20, "readout": "pma", "backbone": "rgat", "num_heads": 2, "color_mlp_hidden_dims": 32, "pma_heads": 4, "pma_sabs": 0, "attn_pool_dropout": 0.0}`
- **rgat/supernode**: `{"gnn_embedding_dim": 128, "gnn_hidden_dim": 156, "k_gnn_layers": 3, "gnn_dropout_rate": 0.0, "gnn_mlp_hidden_dims": 50, "mlp_dropout_rate": 0.1, "weight_decay": 0.001, "feature_norm": "standardize", "k": 20, "readout": "supernode", "backbone": "rgat", "num_heads": 4, "color_mlp_hidden_dims": 32, "supernode_init": "mean", "keep_same_color": true}`
- **rgat/multistat**: `{"gnn_embedding_dim": 128, "gnn_hidden_dim": 100, "k_gnn_layers": 2, "gnn_dropout_rate": 0.2, "gnn_mlp_hidden_dims": 50, "mlp_dropout_rate": 0.0, "weight_decay": 0.01, "feature_norm": "standardize", "k": 20, "readout": "multistat", "backbone": "rgat", "num_heads": 4, "color_mlp_hidden_dims": 50}`
