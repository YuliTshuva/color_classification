# Experiment 09 - learned pooling (pregnancy-learned)

Backbone frozen at the exp08 winner (two-relation graph, MiniLM features, no color
embedding). Configurations selected on validation AUC only; test AUC is never
optimized on. Test AUC is the mean ± standard deviation over 5 seeds (only model init and dropout vary; the split is fixed).

| Backbone | Readout | What was done | Configs | Val AUC (node/color) | Test AUC node | Test AUC color |
|---|---|---|---|---|---|---|
| rgat | `pma` | 1.4 Pooling by Multihead Attention, optionally after self-attention among the members | 45 | 0.764 / 0.806 | 0.677 ± 0.027 | 0.679 ± 0.029 |
| rgat | `multistat` | 1.3 Multi-statistic readout: mean || max || sum concatenated before the color MLP | 35 | 0.791 / 0.768 | 0.695 ± 0.018 | 0.675 ± 0.008 |
| rgat | `node_score_mean` | Baseline. Color score is the mean of the node head's sigmoid scores; no color head | 38 | 0.713 / 0.725 | 0.654 ± 0.035 | 0.658 ± 0.028 |
| rgat | `gated_attn` | 1.1 Gated attention (MIL) pooling; scores softmax-normalized within each color | 39 | 0.768 / 0.793 | 0.641 ± 0.033 | 0.651 ± 0.035 |
| rgat | `mean` | Control (exp08). Parameter-free mean of the member node embeddings, then a color MLP | 36 | 0.794 / 0.768 | 0.656 ± 0.024 | 0.650 ± 0.016 |
| rgat | `supernode` | 1.2 One virtual node per color as a third relation; its embedding feeds the color head | 44 | 0.739 / 0.785 | 0.618 ± 0.012 | 0.622 ± 0.017 |

## Best configuration per readout

- **rgat/pma**: `{"gnn_embedding_dim": 100, "gnn_hidden_dim": 156, "k_gnn_layers": 2, "gnn_dropout_rate": 0.1, "gnn_mlp_hidden_dims": 50, "mlp_dropout_rate": 0.2, "weight_decay": 0.0001, "k": 20, "readout": "pma", "backbone": "rgat", "node_input": "learned", "node_input_dim": 128, "num_heads": 4, "color_mlp_hidden_dims": 50, "pma_heads": 2, "pma_sabs": 0, "attn_pool_dropout": 0.2}`
- **rgat/multistat**: `{"gnn_embedding_dim": 100, "gnn_hidden_dim": 156, "k_gnn_layers": 3, "gnn_dropout_rate": 0.1, "gnn_mlp_hidden_dims": 64, "mlp_dropout_rate": 0.0, "weight_decay": 0.001, "k": 20, "readout": "multistat", "backbone": "rgat", "node_input": "learned", "node_input_dim": 128, "num_heads": 4, "color_mlp_hidden_dims": 50}`
- **rgat/node_score_mean**: `{"gnn_embedding_dim": 156, "gnn_hidden_dim": 100, "k_gnn_layers": 3, "gnn_dropout_rate": 0.0, "gnn_mlp_hidden_dims": 50, "mlp_dropout_rate": 0.2, "weight_decay": 0.0, "k": 20, "readout": "node_score_mean", "backbone": "rgat", "node_input": "learned", "node_input_dim": 128, "num_heads": 4}`
- **rgat/gated_attn**: `{"gnn_embedding_dim": 128, "gnn_hidden_dim": 128, "k_gnn_layers": 3, "gnn_dropout_rate": 0.1, "gnn_mlp_hidden_dims": 50, "mlp_dropout_rate": 0.0, "weight_decay": 0.0001, "k": 20, "readout": "gated_attn", "backbone": "rgat", "node_input": "learned", "node_input_dim": 128, "num_heads": 4, "color_mlp_hidden_dims": 50, "attn_pool_hidden_dim": 128, "attn_pool_dropout": 0.0}`
- **rgat/mean**: `{"gnn_embedding_dim": 100, "gnn_hidden_dim": 100, "k_gnn_layers": 3, "gnn_dropout_rate": 0.1, "gnn_mlp_hidden_dims": 80, "mlp_dropout_rate": 0.0, "weight_decay": 0.0, "k": 20, "readout": "mean", "backbone": "rgat", "node_input": "learned", "node_input_dim": 128, "num_heads": 4, "color_mlp_hidden_dims": 50}`
- **rgat/supernode**: `{"gnn_embedding_dim": 156, "gnn_hidden_dim": 156, "k_gnn_layers": 3, "gnn_dropout_rate": 0.0, "gnn_mlp_hidden_dims": 64, "mlp_dropout_rate": 0.2, "weight_decay": 0.001, "k": 20, "readout": "supernode", "backbone": "rgat", "node_input": "learned", "node_input_dim": 128, "num_heads": 2, "color_mlp_hidden_dims": 50, "supernode_init": "zeros", "keep_same_color": false}`
