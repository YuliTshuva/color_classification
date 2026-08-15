"""
Yuli Tshuva.
train09.py - Channel 1 of the post-exp08 proposals: learned pooling.

The backbone is frozen at the exp08 winner (two-relation graph = kNN edges +
same-color edges, relational GNN, MiniLM features as input, no color embedding),
so the only thing that varies between runs is how node embeddings become a color
prediction. Everything else - data, split, loss, early stopping, evaluation - is
inherited from train08 unchanged, which is what makes the numbers comparable to
Table 1 of the exp08 report.

Readouts:
  mean            : exp08 control. Parameter-free mean of the member node
                    embeddings -> color MLP. Reproduces the rgat/rgcn row.
  node_score_mean : Simplest possible baseline. The color score is the mean of
                    the *node head's* sigmoid scores over its members - no color
                    head at all, zero extra parameters. Says how much of the
                    color performance is just "average your node predictions".
  gated_attn      : Proposal 1.1. Gated attention (MIL) pooling.
  supernode       : Proposal 1.2. One virtual node per color as a third
                    relation; the GNN learns the aggregation itself and the
                    supernode embedding feeds the color head directly.
  multistat       : Proposal 1.3. mean || max || sum concatenated.
  pma             : Proposal 1.4. Pooling by Multihead Attention, optionally
                    preceded by self-attention blocks among the member nodes.

Usage (server), one process per readout:
  python train09.py --readout mean            --trials 60
  python train09.py --readout node_score_mean --trials 60
  python train09.py --readout gated_attn      --trials 60
  python train09.py --readout supernode       --trials 60
  python train09.py --readout multistat       --trials 60
  python train09.py --readout pma             --trials 60
  python train09.py --readout gated_attn --smoke          # quick sanity run

Error bars, after the studies are done:
  python train09.py --readout gated_attn --seed-eval --seeds 5

Results:
  results/exp09/<dataset>/<backbone>_<readout>/trial_results.csv
  results/exp09/<dataset>/<backbone>_<readout>/hp_results.json
  results/exp09/<dataset>/seed_results.csv
"""

# Imports
from models import *
from utils import load_pregnancy_data, load_disease_data
from train08 import (DISEASES, EPOCHS, LR, TOLERANCE, SEED, set_seed, normalize_features,
                     build_same_color_edges, load_pregnancy_node_features, load_disease_node_features)
from sklearn.metrics import roc_auc_score
import argparse
import csv
import json
import optuna
import os
import pandas as pd

READOUTS = ["mean", "node_score_mean", "gated_attn", "supernode", "multistat", "pma"]
BACKBONES = ["rgat", "rgcn"]


def build_supernode_edges(color_indices, n_nodes):
    """
    Proposal 1.2's third relation: one virtual node per color, connected to every
    one of its members. Supernode of color c is given index n_nodes + c, so the
    real nodes keep indices 0..n_nodes-1 and all the existing masks stay valid.
    :return: edge_index of shape (2, n_nodes)
    """
    return torch.stack([torch.arange(n_nodes, dtype=torch.long),
                        n_nodes + color_indices.to(torch.long)])


def build_within_color_pairs(color_indices, same_color_edges, n_nodes):
    """
    Full block-diagonal attention support for the PMA readout's SAB blocks:
    every within-color pair in both directions, plus self-loops.
    :return: edge_index of shape (2, 2 * num_same_color_edges + n_nodes)
    """
    self_loops = torch.arange(n_nodes, dtype=torch.long).repeat(2, 1)
    return torch.cat([same_color_edges, same_color_edges.flip(0), self_loops], dim=1)


def make_readout(readout, dim, hps):
    """Build the pooling module. Returns None for the two headless readouts."""
    if readout == "mean":
        return MeanReadout(dim=dim)
    if readout == "multistat":
        return MultiStatReadout(dim=dim)
    if readout == "gated_attn":
        return GatedAttentionReadout(dim=dim, hidden_dim=hps["attn_pool_hidden_dim"],
                                     dropout_rate=hps["attn_pool_dropout"])
    if readout == "pma":
        return PMAReadout(dim=dim, num_heads=hps["pma_heads"], num_sabs=hps["pma_sabs"],
                          dropout_rate=hps["attn_pool_dropout"])
    if readout in ("supernode", "node_score_mean"):
        return None  # the color representation does not come from a pooling module
    raise ValueError(f"Unknown readout: {readout}")


def train(readout, backbone, n_colors, data, hps, features, epochs=EPOCHS, seed=SEED):
    """
    Train one configuration. Mirrors train08.train, with the variant axis
    replaced by the readout axis.
    :param data: (edge_index, color_indices, color_labels, test_colors, split)
    :param features: [num_nodes, feat_dim] MiniLM embeddings.
    :return: (test_gnn_auc, val_gnn_auc, train_gnn_auc,
              test_color_auc, val_color_auc, train_color_auc)
    """
    set_seed(seed)

    edge_index, color_indices, color_labels, test_colors, split = data
    color_labels = color_labels.float()
    node_labels = color_labels[color_indices].float()
    n_nodes = color_indices.size(0)

    # --- Split non-test colors 80/20 into train and validation (identical to train06/07/08)
    all_colors = torch.arange(n_colors)
    train_val_colors = all_colors[~torch.isin(all_colors, test_colors)]
    n_val = max(1, int(len(train_val_colors) * 0.2))
    perm = torch.randperm(len(train_val_colors), generator=torch.Generator().manual_seed(42))
    val_colors = train_val_colors[perm[:n_val]]
    train_colors = train_val_colors[perm[n_val:]]
    training_mask = torch.isin(color_indices, train_colors)
    val_mask = torch.isin(color_indices, val_colors)
    test_mask = torch.isin(color_indices, test_colors)

    # --- Inputs
    feat = normalize_features(features, split == 0, hps["feature_norm"])
    feat_dim = feat.size(1)

    # --- Relational graph. Relation 0: kNN edges, relation 1: same-color edges.
    # The RGCN/RGAT layers symmetrize internally (directed=False), so each pair
    # is supplied once.
    same_color_edges = build_same_color_edges(color_indices)
    supernode = readout == "supernode"
    # With supernodes the same-color relation is optionally dropped: the
    # supernode already connects the members at two hops, so relation 1 may be
    # redundant. Which of the two wins is an empirical question - hence the hp.
    keep_same_color = (not supernode) or hps["keep_same_color"]

    rel_edges = [edge_index]
    rel_types = [torch.zeros(edge_index.size(1), dtype=torch.long)]
    if keep_same_color:
        rel_edges.append(same_color_edges)
        rel_types.append(torch.ones(same_color_edges.size(1), dtype=torch.long))
    num_relations = len(rel_edges)

    if supernode:
        supernode_edges = build_supernode_edges(color_indices, n_nodes)
        rel_edges.append(supernode_edges)
        rel_types.append(torch.full((supernode_edges.size(1),), num_relations, dtype=torch.long))
        num_relations += 1

        # Input features of the virtual nodes
        if hps["supernode_init"] == "mean":
            super_feat = scatter_mean(feat, color_indices, n_colors)
        else:
            super_feat = torch.zeros(n_colors, feat_dim)
        feat = torch.cat([feat, super_feat], dim=0)

    rel_edge_index = torch.cat(rel_edges, dim=1)
    rel_edge_type = torch.cat(rel_types)

    pair_edge_index = None
    if readout == "pma" and hps["pma_sabs"] > 0:
        pair_edge_index = build_within_color_pairs(color_indices, same_color_edges, n_nodes)

    # --- Modules
    modules = {}
    if backbone == "rgcn":
        modules["gnn"] = RGCN(in_dim=feat_dim, hidden_dim=hps["gnn_hidden_dim"],
                              out_dim=hps["gnn_embedding_dim"], num_relations=num_relations,
                              num_layers=hps["k_gnn_layers"], dropout=hps["gnn_dropout_rate"])
    else:
        modules["gnn"] = RGAT(in_dim=feat_dim, hidden_dim=hps["gnn_hidden_dim"],
                              out_dim=hps["gnn_embedding_dim"], num_relations=num_relations,
                              num_layers=hps["k_gnn_layers"], num_heads=hps["num_heads"],
                              dropout=hps["gnn_dropout_rate"], attn_dropout=hps["gnn_dropout_rate"])

    modules["node_mlp"] = MLP(input_dim=hps["gnn_embedding_dim"], hidden_dims=[hps["gnn_mlp_hidden_dims"]],
                              output_dim=1, dropout_rate=hps["mlp_dropout_rate"])

    pooling = make_readout(readout, hps["gnn_embedding_dim"], hps)
    if pooling is not None:
        modules["readout"] = pooling
    if readout != "node_score_mean":
        color_in = hps["gnn_embedding_dim"] if supernode else pooling.out_dim
        modules["color_head"] = MLP(input_dim=color_in, hidden_dims=[hps["color_mlp_hidden_dims"]],
                                    output_dim=1, dropout_rate=hps["mlp_dropout_rate"])

    # --- Move everything to the device
    for m in modules.values():
        m.to(DEVICE)
        m.train()
    feat = feat.to(DEVICE)
    rel_edge_index, rel_edge_type = rel_edge_index.to(DEVICE), rel_edge_type.to(DEVICE)
    if pair_edge_index is not None:
        pair_edge_index = pair_edge_index.to(DEVICE)
    color_indices = color_indices.to(DEVICE)
    color_labels, node_labels = color_labels.to(DEVICE), node_labels.to(DEVICE)
    training_mask, val_mask, test_mask = training_mask.to(DEVICE), val_mask.to(DEVICE), test_mask.to(DEVICE)
    train_colors, val_colors, test_colors = train_colors.to(DEVICE), val_colors.to(DEVICE), test_colors.to(DEVICE)

    # node_score_mean produces a probability rather than a logit, so it needs the
    # plain BCE; every other readout stays on BCEWithLogits.
    color_is_prob = readout == "node_score_mean"
    classification_loss = nn.BCEWithLogitsLoss()
    params = [p for m in modules.values() for p in m.parameters()]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=hps["weight_decay"])

    def color_loss_fn(out, colors):
        if color_is_prob:
            return F.binary_cross_entropy(out[colors].clamp(1e-6, 1 - 1e-6), color_labels[colors])
        return classification_loss(out[colors], color_labels[colors])

    def forward():
        h = modules["gnn"](feat, rel_edge_index, rel_edge_type)

        # With supernodes the tail rows of h are the virtual nodes; the node task
        # only ever sees the real ones.
        h_nodes = h[:n_nodes] if supernode else h
        node_logits = modules["node_mlp"](h_nodes).squeeze(-1)

        if readout == "node_score_mean":
            node_probs = torch.sigmoid(node_logits).unsqueeze(1)
            color_out = scatter_mean(node_probs, color_indices, n_colors).squeeze(-1)
        elif supernode:
            color_out = modules["color_head"](h[n_nodes:]).squeeze(-1)
        elif readout == "pma":
            pooled = modules["readout"](h_nodes, color_indices, n_colors, pair_edge_index)
            color_out = modules["color_head"](pooled).squeeze(-1)
        else:
            pooled = modules["readout"](h_nodes, color_indices, n_colors)
            color_out = modules["color_head"](pooled).squeeze(-1)

        return node_logits, color_out

    # --- Training loop with early stopping on validation loss (identical to train08)
    best_val_loss, unimproved_epochs = torch.inf, 0
    best_states = None

    for epoch in range(epochs):
        node_logits, color_out = forward()

        loss_gnn = classification_loss(node_logits[training_mask], node_labels[training_mask])
        loss_color = color_loss_fn(color_out, train_colors)
        loss = loss_gnn + loss_color

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_loss = (classification_loss(node_logits[val_mask].detach(), node_labels[val_mask]) +
                        color_loss_fn(color_out.detach(), val_colors)).item()

        if val_loss < best_val_loss * 0.99:
            best_val_loss = val_loss
            unimproved_epochs = 0
            best_states = {name: {k: v.detach().clone() for k, v in m.state_dict().items()}
                           for name, m in modules.items()}
        else:
            unimproved_epochs += 1
        if unimproved_epochs >= TOLERANCE:
            break

    # --- Evaluate with the best checkpoint
    if best_states is not None:
        for name, m in modules.items():
            m.load_state_dict(best_states[name])
    for m in modules.values():
        m.eval()

    with torch.no_grad():
        node_logits, color_out = forward()
    node_scores = torch.sigmoid(node_logits)
    color_scores = color_out if color_is_prob else torch.sigmoid(color_out)

    node_labels = node_labels.long()
    train_gnn_auc = roc_auc_score(node_labels[training_mask].cpu().numpy(), node_scores[training_mask].cpu().numpy())
    val_gnn_auc = roc_auc_score(node_labels[val_mask].cpu().numpy(), node_scores[val_mask].cpu().numpy())
    test_gnn_auc = roc_auc_score(node_labels[test_mask].cpu().numpy(), node_scores[test_mask].cpu().numpy())
    train_color_auc = roc_auc_score(color_labels[train_colors].cpu().numpy(), color_scores[train_colors].cpu().numpy())
    val_color_auc = roc_auc_score(color_labels[val_colors].cpu().numpy(), color_scores[val_colors].cpu().numpy())
    test_color_auc = roc_auc_score(color_labels[test_colors].cpu().numpy(), color_scores[test_colors].cpu().numpy())

    return test_gnn_auc, val_gnn_auc, train_gnn_auc, test_color_auc, val_color_auc, train_color_auc


def sample_hps(trial, readout, backbone, k_range):
    """
    Backbone ranges are exactly train08's (non-wide) space, so a `mean` study
    here reproduces the exp08 rgat/rgcn study. Only the readout-specific
    hyperparameters are new.
    """
    hps = {
        "gnn_embedding_dim": trial.suggest_categorical("gnn_embedding_dim", [100, 128, 156]),
        "gnn_hidden_dim": trial.suggest_categorical("gnn_hidden_dim", [100, 128, 156]),
        "k_gnn_layers": trial.suggest_categorical("k_gnn_layers", [1, 2, 3]),
        "gnn_dropout_rate": trial.suggest_categorical("gnn_dropout_rate", [0.0, 0.1, 0.2]),
        "gnn_mlp_hidden_dims": trial.suggest_categorical("gnn_mlp_hidden_dims", [50, 64, 80]),
        "mlp_dropout_rate": trial.suggest_categorical("mlp_dropout_rate", [0.0, 0.1, 0.2]),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3, 1e-2]),
        "feature_norm": trial.suggest_categorical("feature_norm", ["none", "l2", "standardize"]),
        "k": trial.suggest_categorical("k", k_range),
        "readout": readout,
        "backbone": backbone,
    }
    if backbone == "rgat":
        hps["num_heads"] = trial.suggest_categorical("num_heads", [2, 4])
    if readout != "node_score_mean":
        hps["color_mlp_hidden_dims"] = trial.suggest_categorical("color_mlp_hidden_dims", [32, 50, 64])
    if readout == "gated_attn":
        hps["attn_pool_hidden_dim"] = trial.suggest_categorical("attn_pool_hidden_dim", [32, 64, 128])
        hps["attn_pool_dropout"] = trial.suggest_categorical("attn_pool_dropout", [0.0, 0.1, 0.2])
    if readout == "pma":
        hps["pma_heads"] = trial.suggest_categorical("pma_heads", [2, 4])
        # 0 = plain PMA; 1 = one self-attention block among the members first.
        hps["pma_sabs"] = trial.suggest_categorical("pma_sabs", [0, 1])
        hps["attn_pool_dropout"] = trial.suggest_categorical("attn_pool_dropout", [0.0, 0.1, 0.2])
    if readout == "supernode":
        hps["supernode_init"] = trial.suggest_categorical("supernode_init", ["zeros", "mean"])
        hps["keep_same_color"] = trial.suggest_categorical("keep_same_color", [True, False])
    return hps


def load_all(dataset, dataset_names, k_range):
    """Load every (dataset, k) graph and its node features once, up front."""
    datasets, features = {}, {}
    for name in dataset_names:
        datasets[name] = {}
        for k in k_range:
            if dataset == "diseases":
                edge_index, color_indices, labels, split = load_disease_data(disease=name, k=k)
            else:
                edge_index, color_indices, labels, split = load_pregnancy_data(k=k)
            test_colors = color_indices[split == 1].unique()
            datasets[name][k] = (edge_index, color_indices, labels, test_colors, split, len(labels))
        features[name] = (load_disease_node_features(name) if dataset == "diseases"
                          else load_pregnancy_node_features())
        assert features[name].size(0) == datasets[name][k_range[0]][1].size(0), \
            f"Feature/node count mismatch for {name}"
    return datasets, features


FIELDNAMES = ["readout", "backbone", "trial", "disease",
              "train_gnn_auc", "val_gnn_auc", "test_gnn_auc",
              "train_color_auc", "val_color_auc", "test_color_auc", "hps"]


def append_row(csv_path, row):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def results_dir_of(dataset, backbone, readout):
    return join("results", "exp09", dataset, f"{backbone}_{readout}")


def run_study(dataset, readout, backbone, trials, k_range, diseases, epochs=EPOCHS):
    results_dir = results_dir_of(dataset, backbone, readout)
    os.makedirs(results_dir, exist_ok=True)
    csv_path = join(results_dir, "trial_results.csv")
    hp_path = join(results_dir, "hp_results.json")

    hp_tried = {}
    if os.path.exists(hp_path):
        with open(hp_path, "r") as f:
            hp_tried = json.load(f)

    dataset_names = diseases if dataset == "diseases" else ["pregnancy"]
    datasets, features = load_all(dataset, dataset_names, k_range)

    study = optuna.create_study(direction="maximize")

    def objective(trial):
        hps = sample_hps(trial, readout, backbone, k_range)
        if str(hps) in hp_tried:
            return hp_tried[str(hps)]

        print(f"Trial {trial.number}: {hps}")
        total_auc = 0

        for name in dataset_names:
            edge_index, color_indices, labels, test_colors, split, n_colors = datasets[name][hps["k"]]
            try:
                (test_gnn_auc, val_gnn_auc, train_gnn_auc,
                 test_color_auc, val_color_auc, train_color_auc) = train(
                    readout, backbone, n_colors,
                    data=(edge_index, color_indices, labels, test_colors, split),
                    hps=hps, features=features[name], epochs=epochs)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                # OOM: record the config so it is never retried, prune the trial
                torch.cuda.empty_cache()
                print(f"OOM on {name} - pruning trial {trial.number}")
                hp_tried[str(hps)] = 0.0
                with open(hp_path, "w") as f:
                    json.dump(hp_tried, f, indent=4)
                raise optuna.TrialPruned()

            # Selection on validation AUC only; test AUC is logged, never optimized on
            total_auc += (val_gnn_auc + val_color_auc) / 2

            append_row(csv_path, {"readout": readout, "backbone": backbone, "trial": trial.number, "disease": name,
                                  "train_gnn_auc": round(train_gnn_auc, 4), "val_gnn_auc": round(val_gnn_auc, 4),
                                  "test_gnn_auc": round(test_gnn_auc, 4),
                                  "train_color_auc": round(train_color_auc, 4),
                                  "val_color_auc": round(val_color_auc, 4),
                                  "test_color_auc": round(test_color_auc, 4), "hps": json.dumps(hps)})

        total_auc /= len(dataset_names)
        hp_tried[str(hps)] = total_auc
        with open(hp_path, "w") as f:
            json.dump(hp_tried, f, indent=4)

        print(f"Mean val AUC: {total_auc:.4f}")
        print("*" * 80, "\n")
        return total_auc

    study.optimize(objective, n_trials=trials)


def best_hps_of(dataset, backbone, readout):
    """Pick the study's best config (by mean validation AUC) from its trial CSV."""
    df = pd.read_csv(join(results_dir_of(dataset, backbone, readout), "trial_results.csv"))
    df["val_score"] = (df["val_gnn_auc"] + df["val_color_auc"]) / 2
    return json.loads(df.groupby("hps")["val_score"].mean().idxmax())


def run_seed_eval(dataset, readout, backbone, seeds, diseases, epochs=EPOCHS):
    """Retrain the study's best config over several seeds, for error bars."""
    csv_path = join("results", "exp09", dataset, "seed_results.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    hps = best_hps_of(dataset, backbone, readout)
    print(f"== {backbone}/{readout}: best hps {hps}")

    dataset_names = diseases if dataset == "diseases" else ["pregnancy"]
    datasets, features = load_all(dataset, dataset_names, [hps["k"]])

    for name in dataset_names:
        edge_index, color_indices, labels, test_colors, split, n_colors = datasets[name][hps["k"]]
        for seed in range(seeds):
            (test_gnn_auc, val_gnn_auc, train_gnn_auc,
             test_color_auc, val_color_auc, train_color_auc) = train(
                readout, backbone, n_colors,
                data=(edge_index, color_indices, labels, test_colors, split),
                hps=hps, features=features[name], epochs=epochs, seed=seed)

            append_row(csv_path, {"readout": readout, "backbone": backbone, "trial": seed, "disease": name,
                                  "train_gnn_auc": round(train_gnn_auc, 4), "val_gnn_auc": round(val_gnn_auc, 4),
                                  "test_gnn_auc": round(test_gnn_auc, 4),
                                  "train_color_auc": round(train_color_auc, 4),
                                  "val_color_auc": round(val_color_auc, 4),
                                  "test_color_auc": round(test_color_auc, 4), "hps": json.dumps(hps)})
            print(f"{readout} seed {seed} [{name}]: "
                  f"test gnn {test_gnn_auc:.4f}, test color {test_color_auc:.4f}")

    print(f"Done. Results in {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Run the learned-pooling (Channel 1) experiments.")
    parser.add_argument("--readout", required=True, choices=READOUTS)
    parser.add_argument("--backbone", type=str, default="rgat", choices=BACKBONES)
    parser.add_argument("--dataset", type=str, default="pregnancy", choices=["diseases", "pregnancy"])
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--k", type=int, nargs="+", default=[20])
    parser.add_argument("--diseases", type=str, nargs="+", default=DISEASES)
    parser.add_argument("--seed-eval", action="store_true",
                        help="Skip the study; retrain its best config over --seeds seeds")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Epoch cap. The default relies on early stopping; lower it to rehearse.")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick sanity run: 1 dataset, 1 trial, few epochs")
    args = parser.parse_args()

    if args.seed_eval:
        run_seed_eval(args.dataset, args.readout, args.backbone, args.seeds, args.diseases,
                      epochs=args.epochs)
    elif args.smoke:
        run_study(args.dataset, args.readout, args.backbone, trials=1, k_range=args.k,
                  diseases=args.diseases[:1], epochs=15)
    else:
        run_study(args.dataset, args.readout, args.backbone, trials=args.trials, k_range=args.k,
                  diseases=args.diseases, epochs=args.epochs)


if __name__ == "__main__":
    main()
