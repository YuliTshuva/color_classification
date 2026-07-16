"""
Yuli Tshuva.
train08.py - Experiment runner comparing color-classification variants, designed
to run on servers. Each (dataset, variant) pair is a separate Optuna study whose
results are written under results/exp08/<dataset>/<variant>/ so they can be
exported and analyzed later.

Variants:
  attention  : GNN on color embeddings (node task) + AttentionClassifier applied
               to the color-embedding matrix (color task). Same as train07's
               corrected attention baseline - the control for the comparison.
  pool       : Same node task, but the attention classifier is dropped. Color
               logits = MLP over the mean-pooled GNN node embeddings per color.
  feat       : GNN input = MiniLM node features from the data (no color
               embedding at all); pooled color head.
  feat_color : Direction 1 - GNN input = [node features || color embedding];
               pooled color head.
  rgcn       : Direction 2 - RGCN with 2 relations (kNN edges + same-color
               edges), node features as input, no color embedding; pooled head.
  rgat       : Same graph/relations as rgcn but with relational attention.

Usage (server), one process per variant:
  python train08.py --variant attention  --trials 100
  python train08.py --variant pool       --trials 100
  python train08.py --variant feat       --trials 100
  python train08.py --variant feat_color --trials 100
  python train08.py --variant rgcn       --trials 100
  python train08.py --variant rgat       --trials 100
  python train08.py --variant pool --dataset pregnancy --trials 100
  python train08.py --variant rgcn --smoke               # quick sanity run

Results:
  results/exp08/<dataset>/<variant>/trial_results.csv  (one row per trial x
                          disease, includes the full hps as a json column)
  results/exp08/<dataset>/<variant>/hp_results.json    (mean val AUC per hps)
"""

# Imports
from models import *
from utils import *
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import argparse
import csv
import json
import optuna
import os
import random

# Hyperparameters (kept identical to train06/train07 for comparability)
EPOCHS = 10000
LR = 0.001
TOLERANCE = 30
SEED = 42
DISEASES = ['acute_and_unspecified_renal_failure', 'cardiac_dysrhythmias', 'congestive_heart_failure_nonhypertensive',
            'coronary_atherosclerosis_and_other_heart_disease', 'disorders_of_lipid_metabolism',
            'essential_hypertension', 'fluid_and_electrolyte_disorders',
            'hypertension_with_complications_and_secondary_hypertension']
VARIANTS = ["attention", "pool", "feat", "feat_color", "rgcn", "rgat"]
FEATURE_VARIANTS = {"feat", "feat_color", "rgcn", "rgat"}
RELATIONAL_VARIANTS = {"rgcn", "rgat"}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_disease_node_features(disease):
    """
    Load the MiniLM embeddings of the nodes, replicating the exact filtering of
    data/tamar_data_to_graph.py so row i of the output corresponds to node i in
    nodes.txt (verified: filtered count and color order match the saved graph).
    :return: torch.FloatTensor of shape [num_nodes, feat_dim]
    """
    folder = join("data", "tamars_datasets", disease)
    train_emb = np.load(join(folder, "train_embeddings.npy"))
    test_emb = np.load(join(folder, "test_embeddings.npy"))
    train_df = pd.read_csv(join(folder, "data", "train", "listfile.csv"))
    test_df = pd.read_csv(join(folder, "data", "test", "listfile.csv"))

    assert len(train_emb) == len(train_df) and len(test_emb) == len(test_df)

    emb = np.vstack([train_emb, test_emb])
    df = pd.concat([train_df, test_df], ignore_index=True)

    # Keep only stays that have an episode2 and an episode1 (same as graph build)
    valid_stays = [el.split("_")[0] for el in df["stay"] if "episode2" in el]
    valid_stays = [stay for stay in valid_stays if f"{stay}_episode1_timeseries.csv" in df["stay"].values]
    df["stay_id"] = df["stay"].apply(lambda x: x.split("_")[0])
    emb = emb[df.index[df["stay_id"].isin(valid_stays)]]

    return torch.tensor(emb, dtype=torch.float)


def load_pregnancy_node_features():
    """
    Pregnancy graph nodes are simply all rows of train then test embeddings
    (no filtering in data/pregnancy_data_to_graph.py).
    :return: torch.FloatTensor of shape [num_nodes, feat_dim]
    """
    folder = join("data", "pregnancy_MiniLM_L6")
    train_emb = np.load(join(folder, "train_embeddings.npy"))
    test_emb = np.load(join(folder, "test_embeddings.npy"))
    return torch.tensor(np.vstack([train_emb, test_emb]), dtype=torch.float)


def build_same_color_edges(color_indices):
    """
    Build the second relation: an edge between every pair of nodes that share a
    color (episodes/visits of the same patient). Each pair appears once; the
    RGCN/RGAT layers symmetrize internally.
    :return: edge_index of shape (2, num_same_color_edges)
    """
    groups = {}
    for node, color in enumerate(color_indices.tolist()):
        groups.setdefault(color, []).append(node)

    src, dst = [], []
    for nodes in groups.values():
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                src.append(nodes[i])
                dst.append(nodes[j])

    if len(src) == 0:
        return torch.empty(2, 0, dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


def symmetrize(edge_index):
    """PyG convs treat edge_index as directed; edges.txt stores each pair once."""
    return torch.cat([edge_index, edge_index.flip(0)], dim=1)


def normalize_features(features, fit_mask, mode):
    """
    :param fit_mask: boolean mask of nodes used to fit statistics (train split
                     only, to avoid test leakage).
    :param mode: "none" | "l2" | "standardize"
    """
    if mode == "none":
        return features
    if mode == "l2":
        return F.normalize(features, p=2, dim=1)
    if mode == "standardize":
        mean = features[fit_mask].mean(dim=0, keepdim=True)
        std = features[fit_mask].std(dim=0, keepdim=True).clamp(min=1e-6)
        return (features - mean) / std
    raise ValueError(f"Unknown feature_norm: {mode}")


def scatter_mean_by_color(node_embeddings, color_indices, n_colors):
    """Mean-pool node embeddings per color. :return: [n_colors, dim]"""
    dim = node_embeddings.size(1)
    pooled = torch.zeros(n_colors, dim, device=node_embeddings.device)
    pooled.index_add_(0, color_indices, node_embeddings)
    counts = torch.bincount(color_indices, minlength=n_colors).clamp(min=1)
    return pooled / counts.unsqueeze(1).float()


def make_gnn(model, in_features, hps):
    if model == "GCN":
        return GCN(in_features=in_features, hidden_dim=hps["gnn_hidden_dim"],
                   out_features=hps["gnn_embedding_dim"], K=hps["k_gnn_layers"],
                   dropout_rate=hps["gnn_dropout_rate"])
    elif model == "GAT":
        return GAT(in_features=in_features, hidden_dim=hps["gnn_hidden_dim"],
                   out_features=hps["gnn_embedding_dim"], K=hps["k_gnn_layers"],
                   dropout_rate=hps["gnn_dropout_rate"], heads=4)
    elif model == "GIN":
        return GIN(in_features=in_features, hidden_dim=hps["gnn_hidden_dim"],
                   out_features=hps["gnn_embedding_dim"], K=hps["k_gnn_layers"],
                   dropout_rate=hps["gnn_dropout_rate"])
    elif model == "SAGE":
        return GraphSAGE(in_features=in_features, hidden_dim=hps["gnn_hidden_dim"],
                         out_features=hps["gnn_embedding_dim"], K=hps["k_gnn_layers"],
                         dropout_rate=hps["gnn_dropout_rate"])
    raise ValueError(f"Invalid model type: {model}")


def train(variant, n_colors, data, hps, features=None, epochs=EPOCHS, seed=SEED):
    """
    Train one configuration of the given variant.
    :param data: (edge_index, color_indices, color_labels, test_colors, split)
    :param features: [num_nodes, feat_dim] MiniLM embeddings (feature variants).
    :param seed: seed for model init / dropout; the train/val color split stays
                 fixed (its generator is seeded 42 independently).
    :return: (test_gnn_auc, val_gnn_auc, train_gnn_auc,
              test_color_auc, val_color_auc, train_color_auc)
    """
    set_seed(seed)

    edge_index, color_indices, color_labels, test_colors, split = data
    color_labels = color_labels.float()
    node_labels = color_labels[color_indices].float()

    # --- Split non-test colors 80/20 into train and validation (same as train06/07)
    all_colors = torch.arange(n_colors)
    train_val_colors = all_colors[~torch.isin(all_colors, test_colors)]
    n_val = max(1, int(len(train_val_colors) * 0.2))
    perm = torch.randperm(len(train_val_colors), generator=torch.Generator().manual_seed(42))
    val_colors = train_val_colors[perm[:n_val]]
    train_colors = train_val_colors[perm[n_val:]]
    training_mask = torch.isin(color_indices, train_colors)
    val_mask = torch.isin(color_indices, val_colors)
    test_mask = torch.isin(color_indices, test_colors)

    # --- Build inputs and graph structure per variant
    modules = {}
    relational = variant in RELATIONAL_VARIANTS

    if variant in FEATURE_VARIANTS:
        feat = normalize_features(features, split == 0, hps["feature_norm"])
        feat_dim = feat.size(1)
    else:
        feat = None

    if variant in ("attention", "pool", "feat_color"):
        modules["color_embedding"] = ColorEmbedding(n_colors=n_colors, embedding_dim=hps["color_embedding_dim"])

    if variant in ("attention", "pool"):
        in_features = hps["color_embedding_dim"]
    elif variant == "feat_color":
        in_features = feat_dim + hps["color_embedding_dim"]
    else:  # feat, rgcn, rgat
        in_features = feat_dim

    if relational:
        # Relation 0: kNN edges, relation 1: same-color (same-patient) edges.
        # RGCN/RGAT layers symmetrize internally (directed=False by default).
        same_color_edges = build_same_color_edges(color_indices)
        rel_edge_index = torch.cat([edge_index, same_color_edges], dim=1)
        rel_edge_type = torch.cat([torch.zeros(edge_index.size(1), dtype=torch.long),
                                   torch.ones(same_color_edges.size(1), dtype=torch.long)])
        if variant == "rgcn":
            modules["gnn"] = RGCN(in_dim=in_features, hidden_dim=hps["gnn_hidden_dim"],
                                  out_dim=hps["gnn_embedding_dim"], num_relations=2,
                                  num_layers=hps["k_gnn_layers"], dropout=hps["gnn_dropout_rate"])
        else:
            modules["gnn"] = RGAT(in_dim=in_features, hidden_dim=hps["gnn_hidden_dim"],
                                  out_dim=hps["gnn_embedding_dim"], num_relations=2,
                                  num_layers=hps["k_gnn_layers"], num_heads=hps["num_heads"],
                                  dropout=hps["gnn_dropout_rate"], attn_dropout=hps["gnn_dropout_rate"])
    else:
        # PyG convs treat edge_index as directed - make the kNN graph undirected
        sym_edge_index = symmetrize(edge_index)
        modules["gnn"] = make_gnn(hps["model"], in_features, hps)

    # Node-level head
    modules["node_mlp"] = MLP(input_dim=hps["gnn_embedding_dim"], hidden_dims=[hps["gnn_mlp_hidden_dims"]],
                              output_dim=1, dropout_rate=hps["mlp_dropout_rate"])

    # Color-level head
    if variant == "attention":
        modules["color_head"] = AttentionClassifier(dim=hps["color_embedding_dim"], num_heads=2, num_classes=1,
                                                    attention_hidden_dims=[hps["attention_hidden_dims"]],
                                                    attention_dropout_rate=hps["attention_dropout_rate"],
                                                    n_layers=hps["n_layers"])
    else:
        modules["color_head"] = MLP(input_dim=hps["gnn_embedding_dim"], hidden_dims=[hps["color_mlp_hidden_dims"]],
                                    output_dim=1, dropout_rate=hps["mlp_dropout_rate"])

    # --- Move everything to the device
    for m in modules.values():
        m.to(DEVICE)
        m.train()
    if feat is not None:
        feat = feat.to(DEVICE)
    if relational:
        rel_edge_index = rel_edge_index.to(DEVICE)
        rel_edge_type = rel_edge_type.to(DEVICE)
    else:
        sym_edge_index = sym_edge_index.to(DEVICE)
    color_indices = color_indices.to(DEVICE)
    color_labels = color_labels.to(DEVICE)
    node_labels = node_labels.to(DEVICE)
    training_mask, val_mask, test_mask = training_mask.to(DEVICE), val_mask.to(DEVICE), test_mask.to(DEVICE)
    train_colors, val_colors, test_colors = train_colors.to(DEVICE), val_colors.to(DEVICE), test_colors.to(DEVICE)

    classification_loss = nn.BCEWithLogitsLoss()
    params = [p for m in modules.values() for p in m.parameters()]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=hps["weight_decay"])

    def forward():
        # Build the node input
        if variant in ("attention", "pool"):
            x = modules["color_embedding"](color_indices)
        elif variant == "feat_color":
            x = torch.cat([feat, modules["color_embedding"](color_indices)], dim=1)
        else:
            x = feat

        # GNN
        if relational:
            h = modules["gnn"](x, rel_edge_index, rel_edge_type)
        else:
            h = modules["gnn"](x, sym_edge_index)

        node_logits = modules["node_mlp"](h).squeeze(-1)

        # Color-level logits
        if variant == "attention":
            # Attention over the color-embedding matrix itself: [n_colors, dim]
            color_logits = modules["color_head"](modules["color_embedding"].color_embedding.weight).squeeze(-1)
        else:
            pooled = scatter_mean_by_color(h, color_indices, n_colors)
            color_logits = modules["color_head"](pooled).squeeze(-1)

        return node_logits, color_logits

    # --- Training loop with early stopping on validation loss (same as train06/07)
    best_val_loss, unimproved_epochs = torch.inf, 0
    best_states = None

    for epoch in range(epochs):
        node_logits, color_logits = forward()

        loss_gnn = classification_loss(node_logits[training_mask], node_labels[training_mask])
        loss_color = classification_loss(color_logits[train_colors], color_labels[train_colors])
        loss = loss_gnn + loss_color

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_loss = (classification_loss(node_logits[val_mask].detach(), node_labels[val_mask]) +
                        classification_loss(color_logits[val_colors].detach(), color_labels[val_colors])).item()

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
        node_logits, color_logits = forward()
    node_scores = torch.sigmoid(node_logits)
    color_scores = torch.sigmoid(color_logits)

    node_labels = node_labels.long()
    train_gnn_auc = roc_auc_score(node_labels[training_mask].cpu().numpy(), node_scores[training_mask].cpu().numpy())
    val_gnn_auc = roc_auc_score(node_labels[val_mask].cpu().numpy(), node_scores[val_mask].cpu().numpy())
    test_gnn_auc = roc_auc_score(node_labels[test_mask].cpu().numpy(), node_scores[test_mask].cpu().numpy())
    train_color_auc = roc_auc_score(color_labels[train_colors].cpu().numpy(), color_scores[train_colors].cpu().numpy())
    val_color_auc = roc_auc_score(color_labels[val_colors].cpu().numpy(), color_scores[val_colors].cpu().numpy())
    test_color_auc = roc_auc_score(color_labels[test_colors].cpu().numpy(), color_scores[test_colors].cpu().numpy())

    return test_gnn_auc, val_gnn_auc, train_gnn_auc, test_color_auc, val_color_auc, train_color_auc


def sample_hps(trial, variant, models, k_range, wide=False):
    """
    Sample the hyperparameters relevant to the given variant (same ranges as
    train06/07). With wide=True the ranges are expanded upward around the best
    exp08 rgat/rgcn configs, which sat at the boundary of the original space
    (dim 156, 3 layers, 4 heads). All wide dims are divisible by 8 so any head
    count is valid.
    """
    if wide:
        hps = {
            "gnn_embedding_dim": trial.suggest_categorical("gnn_embedding_dim", [128, 160, 200, 256]),
            "gnn_hidden_dim": trial.suggest_categorical("gnn_hidden_dim", [128, 160, 200, 256]),
            "k_gnn_layers": trial.suggest_categorical("k_gnn_layers", [2, 3, 4, 5]),
            "gnn_dropout_rate": trial.suggest_categorical("gnn_dropout_rate", [0.0, 0.1, 0.2, 0.3]),
            "gnn_mlp_hidden_dims": trial.suggest_categorical("gnn_mlp_hidden_dims", [50, 64, 80, 128]),
            "mlp_dropout_rate": trial.suggest_categorical("mlp_dropout_rate", [0.0, 0.1, 0.2]),
            "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 3e-4, 1e-3, 1e-2]),
            "k": trial.suggest_categorical("k", k_range),
            "variant": variant,
            "hp_space": "wide",
        }
    else:
        hps = {
            "gnn_embedding_dim": trial.suggest_categorical("gnn_embedding_dim", [100, 128, 156]),
            "gnn_hidden_dim": trial.suggest_categorical("gnn_hidden_dim", [100, 128, 156]),
            "k_gnn_layers": trial.suggest_categorical("k_gnn_layers", [1, 2, 3]),
            "gnn_dropout_rate": trial.suggest_categorical("gnn_dropout_rate", [0.0, 0.1, 0.2]),
            "gnn_mlp_hidden_dims": trial.suggest_categorical("gnn_mlp_hidden_dims", [50, 64, 80]),
            "mlp_dropout_rate": trial.suggest_categorical("mlp_dropout_rate", [0.0, 0.1, 0.2]),
            "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3, 1e-2]),
            "k": trial.suggest_categorical("k", k_range),
            "variant": variant,
        }
    if variant in ("attention", "pool", "feat_color"):
        hps["color_embedding_dim"] = trial.suggest_categorical("color_embedding_dim", [12, 16, 20, 24])
    if variant in FEATURE_VARIANTS:
        hps["feature_norm"] = trial.suggest_categorical("feature_norm", ["none", "l2", "standardize"])
    if variant == "attention":
        hps["attention_hidden_dims"] = trial.suggest_categorical("attention_hidden_dims", [4, 6, 8])
        hps["attention_dropout_rate"] = trial.suggest_categorical("attention_dropout_rate", [0.0, 0.1, 0.2])
        hps["n_layers"] = trial.suggest_categorical("n_layers", [1, 2, 3, 4])
    else:
        hps["color_mlp_hidden_dims"] = trial.suggest_categorical("color_mlp_hidden_dims", [32, 50, 64])
    if variant in RELATIONAL_VARIANTS:
        hps["model"] = variant.upper()
        if variant == "rgat":
            hps["num_heads"] = trial.suggest_categorical("num_heads", [2, 4, 8] if wide else [2, 4])
    else:
        hps["model"] = trial.suggest_categorical("model", models)
    return hps


def run_study(dataset, variant, trials, k_range, models, diseases, epochs=EPOCHS, wide=False):
    results_dir = join("results", "exp08", dataset, variant + ("_wide" if wide else ""))
    os.makedirs(results_dir, exist_ok=True)
    csv_path = join(results_dir, "trial_results.csv")
    hp_path = join(results_dir, "hp_results.json")

    hp_tried = {}
    if os.path.exists(hp_path):
        with open(hp_path, "r") as f:
            hp_tried = json.load(f)

    # Load all datasets (and features when needed) in advance
    dataset_names = diseases if dataset == "diseases" else ["pregnancy"]
    datasets, features = {}, {}
    for name in dataset_names:
        datasets[name] = {}
        for k in k_range:
            if dataset == "diseases":
                edge_index, color_indices, labels, split = load_disease_data(disease=name, k=k)
            else:
                edge_index, color_indices, labels, split = load_pregnancy_data(k=k)
            n_colors = len(labels)
            test_colors = color_indices[split == 1].unique()
            datasets[name][k] = (edge_index, color_indices, labels, test_colors, split, n_colors)
        if variant in FEATURE_VARIANTS:
            if dataset == "diseases":
                features[name] = load_disease_node_features(name)
            else:
                features[name] = load_pregnancy_node_features()
            assert features[name].size(0) == datasets[name][k_range[0]][1].size(0), \
                f"Feature/node count mismatch for {name}"

    study = optuna.create_study(direction="maximize")

    def objective(trial):
        hps = sample_hps(trial, variant, models, k_range, wide=wide)
        if str(hps) in hp_tried:
            return hp_tried[str(hps)]

        print(f"Trial {trial.number}: {hps}")
        total_auc = 0

        for name in dataset_names:
            edge_index, color_indices, labels, test_colors, split, n_colors = datasets[name][hps["k"]]
            try:
                (test_gnn_auc, val_gnn_auc, train_gnn_auc,
                 test_color_auc, val_color_auc, train_color_auc) = train(
                    variant, n_colors,
                    data=(edge_index, color_indices, labels, test_colors, split),
                    hps=hps, features=features.get(name), epochs=epochs)
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

            # Select hyperparameters on validation AUC; test AUC is logged but never optimized on
            total_auc += (val_gnn_auc + val_color_auc) / 2

            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["variant", "trial", "disease",
                                                       "train_gnn_auc", "val_gnn_auc", "test_gnn_auc",
                                                       "train_color_auc", "val_color_auc", "test_color_auc", "hps"])
                if write_header:
                    writer.writeheader()
                writer.writerow({"variant": variant, "trial": trial.number, "disease": name,
                                 "train_gnn_auc": round(train_gnn_auc, 4), "val_gnn_auc": round(val_gnn_auc, 4),
                                 "test_gnn_auc": round(test_gnn_auc, 4), "train_color_auc": round(train_color_auc, 4),
                                 "val_color_auc": round(val_color_auc, 4), "test_color_auc": round(test_color_auc, 4),
                                 "hps": json.dumps(hps)})

        total_auc /= len(dataset_names)
        hp_tried[str(hps)] = total_auc
        with open(hp_path, "w") as f:
            json.dump(hp_tried, f, indent=4)

        print(f"Mean val AUC: {total_auc:.4f}")
        print("*" * 80, "\n")
        return total_auc

    study.optimize(objective, n_trials=trials)


def main():
    parser = argparse.ArgumentParser(description="Run color-classification variant experiments.")
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--dataset", type=str, default="diseases", choices=["diseases", "pregnancy"])
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--k", type=int, nargs="+", default=[20])
    parser.add_argument("--models", type=str, default="GIN",
                        help="Comma-separated GNN types for non-relational variants (GCN,GAT,GIN,SAGE)")
    parser.add_argument("--diseases", type=str, nargs="+", default=DISEASES)
    parser.add_argument("--wide", action="store_true",
                        help="Expanded hp ranges (follow-up around the best exp08 configs); "
                             "results go to <variant>_wide/")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick sanity run: 1 disease, 1 trial, few epochs")
    args = parser.parse_args()

    models = args.models.split(",")
    if args.smoke:
        run_study(args.dataset, args.variant, trials=1, k_range=args.k, models=models,
                  diseases=args.diseases[:1], epochs=15, wide=args.wide)
    else:
        run_study(args.dataset, args.variant, trials=args.trials, k_range=args.k, models=models,
                  diseases=args.diseases, wide=args.wide)


if __name__ == '__main__':
    main()
