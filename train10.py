"""
Yuli Tshuva.
train10.py - Channel 2 of the post-exp08 proposals: graph construction.

Experiment 09 closed Channel 1: no learned readout beat parameter-free
mean-pooling, and a zero-parameter baseline tied every one of them. Whatever the
color task can extract is already in the per-node predictions, so improving the
color task now means improving *node* representations - which the graph controls.
The graph is the last unlearned component of the pipeline.

This experiment varies the kNN relation and nothing else. The readout is frozen
at exp08/exp09's mean-pooling, the backbone hyperparameter space is train09's
verbatim, and the same-color relation is untouched, so the numbers drop straight
into the exp08/exp09 tables.

Graphs:
  knn20  : control. The exp08 graph - cosine kNN at a fixed k (20) read from
           edges.txt. Reproduces the rgat/mean row of exp09.
  hybrid : a k_core-nearest-neighbour core (guarantees a minimum degree, so no
           node is isolated) plus every further candidate whose cosine
           similarity exceeds a threshold tau. Degree adapts to local density
           instead of being pinned at k for every node.

tau is swept as a quantile q of the candidate similarities rather than as a raw
cosine value - see graph_construction.py. q=0 is plain kNN at k_max, q=1 is plain
kNN at k_core, and the exp08 density sits near q = 1 - 20/k_max, so the sweep
brackets the control on both sides. The `split_core_relation` hyperparameter asks
whether the core and threshold edges are worth separating into two relations: the
relational GNN already weights relations independently, so it can decide how much
to trust each.

The second graph axis is `sim_space`. The MiniLM embeddings are severely
anisotropic - the mean cosine between two *random* pregnancy nodes is +0.979, so
the exp08 neighbourhoods are decided by differences in the fourth decimal place.
Centering or whitening removes that common component and changes the top-20
neighbourhoods almost completely (27% and 12% overlap with raw respectively),
which makes the space a bigger lever on the graph than the threshold is. It is
swept rather than assumed: measured on label agreement alone the three spaces are
all within ~1.5 points of chance, so which one helps is an open question.

Selection is on validation AUC only; test AUC is logged and never optimized on.

Prerequisite (once per dataset - do NOT let parallel jobs race on this):
  python graph_construction.py --dataset pregnancy --k-max 50 --verify --describe

Usage (server), one process per graph:
  python train10.py --graph knn20  --trials 60
  python train10.py --graph hybrid --trials 100
  python train10.py --graph hybrid --smoke            # quick sanity run

Featureless ablation (--node-input, see NODE_INPUTS): drops the embeddings from
the GNN input while leaving the kNN graph - which is still built from them -
alone. Asks how much of the score was the features and how much was the graph.
Those runs write to results/exp10/<dataset>-<node input>/, so nothing already
reported is touched:
  python train10.py --graph hybrid --node-input learned

Error bars, after the studies are done:
  python train10.py --graph hybrid --seed-eval --seeds 5

Results:
  results/exp10/<dataset>/<backbone>_<graph>/trial_results.csv
  results/exp10/<dataset>/<backbone>_<graph>/hp_results.json
  results/exp10/<dataset>/seed_results.csv
"""

# Imports
from models import *
from utils import load_pregnancy_data, load_disease_data
from train08 import (DISEASES, EPOCHS, LR, TOLERANCE, SEED, NODE_INPUTS, set_seed, normalize_features,
                     build_same_color_edges, load_pregnancy_node_features, load_disease_node_features,
                     resolve_dataset, resolve_node_input, results_tag, build_node_input)
import constants
from graph_construction import SIM_SPACES, build_hybrid_edges, load_candidates
from sklearn.metrics import roc_auc_score
import argparse
import csv
import json
import optuna
import os
import pandas as pd

GRAPHS = ["knn20", "hybrid"]
BACKBONES = ["rgat", "rgcn"]
K_MAX = 50  # candidate neighbours per node; the ceiling on any threshold graph

# Built once per (name, k_core, quantile) and reused across trials - the sweep
# revisits the same graph many times with different backbone hyperparameters.
_GRAPH_CACHE = {}


def edge_stats(edge_index, n_nodes):
    """Density summary of a single relation, in the format build_hybrid_edges returns."""
    degrees = torch.bincount(edge_index.reshape(-1), minlength=n_nodes)
    return {
        "tau": float("nan"),
        "n_core_edges": edge_index.size(1),
        "n_extra_edges": 0,
        "n_edges": edge_index.size(1),
        "mean_degree": round(float(degrees.float().mean()), 2),
        "max_degree": int(degrees.max()),
        "isolated_nodes": int((degrees == 0).sum()),
        "saturated_frac": float("nan"),
    }


def build_relations(graph, hps, edge_index, same_color_edges, candidates, n_nodes, name):
    """
    Assemble the relational graph. Relation order is kNN-ish relations first,
    same-color last, matching train08/train09 so the learned per-relation weights
    stay comparable.

    :return: (rel_edge_index [2, E], rel_edge_type [E], num_relations, stats)
    """
    if graph == "knn20":
        rel_edges = [edge_index]
        stats = edge_stats(edge_index, n_nodes)
    else:
        key = (name, hps["sim_space"], hps["k_core"], hps["sim_quantile"])
        if key not in _GRAPH_CACHE:
            nbr_idx, nbr_sim = candidates[hps["sim_space"]]
            _GRAPH_CACHE[key] = build_hybrid_edges(nbr_idx, nbr_sim,
                                                   k_core=hps["k_core"],
                                                   quantile=hps["sim_quantile"])
        core_edges, extra_edges, stats = _GRAPH_CACHE[key]

        if hps["split_core_relation"]:
            rel_edges = [core_edges, extra_edges]
        else:
            rel_edges = [torch.cat([core_edges, extra_edges], dim=1)]

    rel_edges.append(same_color_edges)
    rel_types = [torch.full((e.size(1),), r, dtype=torch.long) for r, e in enumerate(rel_edges)]

    return torch.cat(rel_edges, dim=1), torch.cat(rel_types), len(rel_edges), stats


def train(graph, backbone, n_colors, data, hps, features, candidates, name,
          epochs=EPOCHS, seed=SEED):
    """
    Train one configuration. Mirrors train09.train with the readout frozen at
    `mean` and the graph as the varying axis.
    :param data: (edge_index, color_indices, color_labels, test_colors, split,
                  same_color_edges)
    :param features: [num_nodes, feat_dim] MiniLM embeddings.
    :param candidates: {sim_space: (nbr_idx, nbr_sim)} from graph_construction;
                       None for knn20.
    :return: (metrics tuple, graph stats dict)
    """
    set_seed(seed)

    edge_index, color_indices, color_labels, test_colors, split, same_color_edges = data
    color_labels = color_labels.float()
    node_labels = color_labels[color_indices].float()
    n_nodes = color_indices.size(0)

    # --- Split non-test colors 80/20 into train and validation (identical to train06-09)
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
    feat, feat_module, feat_dim = build_node_input(features, split, hps, n_nodes)

    # --- Relational graph. The RGCN/RGAT layers symmetrize internally
    # (directed=False), so each pair is supplied once.
    rel_edge_index, rel_edge_type, num_relations, graph_stats = build_relations(
        graph, hps, edge_index, same_color_edges, candidates, n_nodes, name)

    # --- Modules
    modules = {}
    if feat_module is not None:
        modules["node_input"] = feat_module
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
    modules["readout"] = MeanReadout(dim=hps["gnn_embedding_dim"])
    modules["color_head"] = MLP(input_dim=hps["gnn_embedding_dim"], hidden_dims=[hps["color_mlp_hidden_dims"]],
                                output_dim=1, dropout_rate=hps["mlp_dropout_rate"])

    # --- Move everything to the device
    for m in modules.values():
        m.to(DEVICE)
        m.train()
    if feat is not None:
        feat = feat.to(DEVICE)
    rel_edge_index, rel_edge_type = rel_edge_index.to(DEVICE), rel_edge_type.to(DEVICE)
    color_indices = color_indices.to(DEVICE)
    color_labels, node_labels = color_labels.to(DEVICE), node_labels.to(DEVICE)
    training_mask, val_mask, test_mask = training_mask.to(DEVICE), val_mask.to(DEVICE), test_mask.to(DEVICE)
    train_colors, val_colors, test_colors = train_colors.to(DEVICE), val_colors.to(DEVICE), test_colors.to(DEVICE)

    classification_loss = nn.BCEWithLogitsLoss()
    params = [p for m in modules.values() for p in m.parameters()]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=hps["weight_decay"])

    def node_x():
        """The GNN input: a fixed tensor, or the trainable table when learned."""
        return feat if feat_module is None else modules["node_input"].weight

    def forward():
        h = modules["gnn"](node_x(), rel_edge_index, rel_edge_type)
        node_logits = modules["node_mlp"](h).squeeze(-1)
        pooled = modules["readout"](h, color_indices, n_colors)
        color_out = modules["color_head"](pooled).squeeze(-1)
        return node_logits, color_out

    # --- Training loop with early stopping on validation loss (identical to train08/09)
    best_val_loss, unimproved_epochs = torch.inf, 0
    best_states = None

    for epoch in range(epochs):
        node_logits, color_out = forward()

        loss_gnn = classification_loss(node_logits[training_mask], node_labels[training_mask])
        loss_color = classification_loss(color_out[train_colors], color_labels[train_colors])
        loss = loss_gnn + loss_color

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_loss = (classification_loss(node_logits[val_mask].detach(), node_labels[val_mask]) +
                        classification_loss(color_out[val_colors].detach(), color_labels[val_colors])).item()

        if val_loss < best_val_loss * 0.99:
            best_val_loss = val_loss
            unimproved_epochs = 0
            best_states = {name_: {k: v.detach().clone() for k, v in m.state_dict().items()}
                           for name_, m in modules.items()}
        else:
            unimproved_epochs += 1
        if unimproved_epochs >= TOLERANCE:
            break

    # --- Evaluate with the best checkpoint
    if best_states is not None:
        for name_, m in modules.items():
            m.load_state_dict(best_states[name_])
    for m in modules.values():
        m.eval()

    with torch.no_grad():
        node_logits, color_out = forward()
    node_scores = torch.sigmoid(node_logits)
    color_scores = torch.sigmoid(color_out)

    node_labels = node_labels.long()
    train_gnn_auc = roc_auc_score(node_labels[training_mask].cpu().numpy(), node_scores[training_mask].cpu().numpy())
    val_gnn_auc = roc_auc_score(node_labels[val_mask].cpu().numpy(), node_scores[val_mask].cpu().numpy())
    test_gnn_auc = roc_auc_score(node_labels[test_mask].cpu().numpy(), node_scores[test_mask].cpu().numpy())
    train_color_auc = roc_auc_score(color_labels[train_colors].cpu().numpy(), color_scores[train_colors].cpu().numpy())
    val_color_auc = roc_auc_score(color_labels[val_colors].cpu().numpy(), color_scores[val_colors].cpu().numpy())
    test_color_auc = roc_auc_score(color_labels[test_colors].cpu().numpy(), color_scores[test_colors].cpu().numpy())

    metrics = (test_gnn_auc, val_gnn_auc, train_gnn_auc, test_color_auc, val_color_auc, train_color_auc)
    return metrics, graph_stats


def sample_hps(trial, graph, backbone, k_range, quantiles, k_cores, sim_spaces):
    """
    The backbone ranges are train09's verbatim, so a `knn20` study here reproduces
    the exp09 rgat/mean study. Only the graph hyperparameters are new.
    """
    hps = {
        "gnn_embedding_dim": trial.suggest_categorical("gnn_embedding_dim", [100, 128, 156]),
        "gnn_hidden_dim": trial.suggest_categorical("gnn_hidden_dim", [100, 128, 156]),
        "k_gnn_layers": trial.suggest_categorical("k_gnn_layers", [1, 2, 3]),
        "gnn_dropout_rate": trial.suggest_categorical("gnn_dropout_rate", [0.0, 0.1, 0.2]),
        "gnn_mlp_hidden_dims": trial.suggest_categorical("gnn_mlp_hidden_dims", [50, 64, 80]),
        "mlp_dropout_rate": trial.suggest_categorical("mlp_dropout_rate", [0.0, 0.1, 0.2]),
        "color_mlp_hidden_dims": trial.suggest_categorical("color_mlp_hidden_dims", [32, 50, 64]),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3, 1e-2]),
        "graph": graph,
        "backbone": backbone,
    }
    if constants.NODE_INPUT == "feat":
        hps["feature_norm"] = trial.suggest_categorical("feature_norm", ["none", "l2", "standardize"])
    else:
        # Recorded so the arm is visible in the logged hps. "feat" is left out on
        # purpose, so the existing hp_results.json caches still match.
        hps["node_input"] = constants.NODE_INPUT
        if constants.NODE_INPUT in ("random", "learned"):
            hps["node_input_dim"] = trial.suggest_categorical("node_input_dim", [32, 64, 128])
    if backbone == "rgat":
        hps["num_heads"] = trial.suggest_categorical("num_heads", [2, 4])

    if graph == "knn20":
        hps["k"] = trial.suggest_categorical("k", k_range)
    else:
        # tau as a quantile of the candidate similarities: 0 -> full k_max graph,
        # 1 -> pure k_core graph. The exp08 density sits near 1 - 20/K_MAX.
        hps["sim_quantile"] = trial.suggest_categorical("sim_quantile", quantiles)
        hps["k_core"] = trial.suggest_categorical("k_core", k_cores)
        hps["split_core_relation"] = trial.suggest_categorical("split_core_relation", [True, False])
        # The space the similarity is measured in. "raw" reproduces the exp08
        # neighbourhoods; the others strip the anisotropic common component.
        hps["sim_space"] = trial.suggest_categorical("sim_space", sim_spaces)
    return hps


def load_all(dataset, dataset_names, k_range, sim_spaces):
    """
    Load every (dataset, k) graph, its node features and its candidates once, up
    front. `sim_spaces` empty means the knn20 control, which needs no candidates.
    """
    datasets, features, candidates = {}, {}, {}
    for name in dataset_names:
        datasets[name] = {}
        same_color_edges = None
        split = None
        for k in k_range:
            if dataset == "diseases":
                edge_index, color_indices, labels, split = load_disease_data(disease=name, k=k)
            else:
                edge_index, color_indices, labels, split = load_pregnancy_data(k=k)
            test_colors = color_indices[split == 1].unique()
            # The same-color relation does not depend on k; building it is a slow
            # Python double loop, so do it once per dataset rather than per trial.
            if same_color_edges is None:
                same_color_edges = build_same_color_edges(color_indices)
            datasets[name][k] = (edge_index, color_indices, labels, test_colors, split,
                                 same_color_edges, len(labels))

        features[name] = (load_disease_node_features(name) if dataset == "diseases"
                          else load_pregnancy_node_features())
        assert features[name].size(0) == datasets[name][k_range[0]][1].size(0), \
            f"Feature/node count mismatch for {name}"

        # Cold caches are built by run_exp10.py's candidate stage, serially, so
        # parallel study jobs never race to write the same file.
        candidates[name] = {
            space: load_candidates(dataset, name, K_MAX, space=space,
                                   features=features[name], fit_mask=(split == 0))
            for space in sim_spaces}
    return datasets, features, candidates


FIELDNAMES = ["graph", "backbone", "trial", "disease",
              "train_gnn_auc", "val_gnn_auc", "test_gnn_auc",
              "train_color_auc", "val_color_auc", "test_color_auc",
              "tau", "n_edges", "mean_degree", "max_degree", "isolated_nodes",
              "saturated_frac", "hps"]


def append_row(csv_path, row):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def make_row(graph, backbone, trial, name, metrics, stats, hps):
    (test_gnn_auc, val_gnn_auc, train_gnn_auc,
     test_color_auc, val_color_auc, train_color_auc) = metrics
    return {"graph": graph, "backbone": backbone, "trial": trial, "disease": name,
            "train_gnn_auc": round(train_gnn_auc, 4), "val_gnn_auc": round(val_gnn_auc, 4),
            "test_gnn_auc": round(test_gnn_auc, 4),
            "train_color_auc": round(train_color_auc, 4), "val_color_auc": round(val_color_auc, 4),
            "test_color_auc": round(test_color_auc, 4),
            "tau": stats["tau"], "n_edges": stats["n_edges"], "mean_degree": stats["mean_degree"],
            "max_degree": stats["max_degree"], "isolated_nodes": stats["isolated_nodes"],
            "saturated_frac": stats["saturated_frac"], "hps": json.dumps(hps)}


def results_dir_of(dataset, backbone, graph):
    return join("results", "exp10", results_tag(dataset), f"{backbone}_{graph}")


def run_study(dataset, graph, backbone, trials, k_range, quantiles, k_cores, sim_spaces,
              diseases, epochs=EPOCHS):
    results_dir = results_dir_of(dataset, backbone, graph)
    os.makedirs(results_dir, exist_ok=True)
    csv_path = join(results_dir, "trial_results.csv")
    hp_path = join(results_dir, "hp_results.json")

    hp_tried = {}
    if os.path.exists(hp_path):
        with open(hp_path, "r") as f:
            hp_tried = json.load(f)

    dataset_names = diseases if dataset == "diseases" else ["pregnancy"]
    datasets, features, candidates = load_all(dataset, dataset_names, k_range,
                                              sim_spaces if graph == "hybrid" else [])

    study = optuna.create_study(direction="maximize")

    def objective(trial):
        hps = sample_hps(trial, graph, backbone, k_range, quantiles, k_cores, sim_spaces)
        if str(hps) in hp_tried:
            return hp_tried[str(hps)]

        print(f"Trial {trial.number}: {hps}")
        total_auc = 0

        for name in dataset_names:
            (edge_index, color_indices, labels, test_colors, split,
             same_color_edges, n_colors) = datasets[name][hps.get("k", k_range[0])]
            try:
                metrics, stats = train(
                    graph, backbone, n_colors,
                    data=(edge_index, color_indices, labels, test_colors, split, same_color_edges),
                    hps=hps, features=features[name], candidates=candidates[name], name=name,
                    epochs=epochs)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                # OOM: record the config so it is never retried, prune the trial.
                # Expected at the dense end of the sweep (low sim_quantile).
                torch.cuda.empty_cache()
                print(f"OOM on {name} - pruning trial {trial.number}")
                hp_tried[str(hps)] = 0.0
                with open(hp_path, "w") as f:
                    json.dump(hp_tried, f, indent=4)
                raise optuna.TrialPruned()

            # Selection on validation AUC only; test AUC is logged, never optimized on
            total_auc += (metrics[1] + metrics[4]) / 2
            append_row(csv_path, make_row(graph, backbone, trial.number, name, metrics, stats, hps))

        total_auc /= len(dataset_names)
        hp_tried[str(hps)] = total_auc
        with open(hp_path, "w") as f:
            json.dump(hp_tried, f, indent=4)

        print(f"Mean val AUC: {total_auc:.4f}")
        print("*" * 80, "\n")
        return total_auc

    study.optimize(objective, n_trials=trials)


def best_hps_of(dataset, backbone, graph):
    """Pick the study's best config (by mean validation AUC) from its trial CSV."""
    df = pd.read_csv(join(results_dir_of(dataset, backbone, graph), "trial_results.csv"))
    df["val_score"] = (df["val_gnn_auc"] + df["val_color_auc"]) / 2
    return json.loads(df.groupby("hps")["val_score"].mean().idxmax())


def run_seed_eval(dataset, graph, backbone, seeds, k_range, diseases, epochs=EPOCHS):
    """Retrain the study's best config over several seeds, for error bars."""
    csv_path = join("results", "exp10", results_tag(dataset), "seed_results.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    hps = best_hps_of(dataset, backbone, graph)
    print(f"== {backbone}/{graph}: best hps {hps}")

    dataset_names = diseases if dataset == "diseases" else ["pregnancy"]
    k = hps.get("k", k_range[0])
    # Only the space the winning config actually used needs loading
    datasets, features, candidates = load_all(dataset, dataset_names, [k],
                                              [hps["sim_space"]] if graph == "hybrid" else [])

    for name in dataset_names:
        (edge_index, color_indices, labels, test_colors, split,
         same_color_edges, n_colors) = datasets[name][k]
        for seed in range(seeds):
            metrics, stats = train(
                graph, backbone, n_colors,
                data=(edge_index, color_indices, labels, test_colors, split, same_color_edges),
                hps=hps, features=features[name], candidates=candidates[name], name=name,
                epochs=epochs, seed=seed)

            append_row(csv_path, make_row(graph, backbone, seed, name, metrics, stats, hps))
            print(f"{graph} seed {seed} [{name}]: "
                  f"test gnn {metrics[0]:.4f}, test color {metrics[3]:.4f}")

    print(f"Done. Results in {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Run the graph-construction (Channel 2) experiments.")
    parser.add_argument("--graph", required=True, choices=GRAPHS)
    parser.add_argument("--backbone", type=str, default="rgat", choices=BACKBONES)
    parser.add_argument("--dataset", type=str, default="pregnancy",
                        help="diseases, pregnancy, or another data/ folder with the pregnancy "
                             "layout (e.g. pregnancy_C for the anti-collapse embeddings)")
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--node-input", type=str, default="feat", choices=NODE_INPUTS,
                        help="What the GNN reads at its input layer. Anything but 'feat' drops the "
                             "embeddings from the input and writes to results/exp10/<dataset>-<node input>/")
    parser.add_argument("--k", type=int, nargs="+", default=[20],
                        help="Prebuilt kNN graphs for the knn20 control; also supplies node metadata.")
    parser.add_argument("--quantiles", type=float, nargs="+",
                        default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        help="Similarity-threshold quantiles to sweep. 0.6 is roughly the k=20 density.")
    parser.add_argument("--k-cores", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--sim-spaces", type=str, nargs="+", default=SIM_SPACES, choices=SIM_SPACES,
                        help="Space the similarity is measured in. Pass just 'raw' to sweep the "
                             "threshold alone, on the exp08 neighbourhoods.")
    parser.add_argument("--diseases", type=str, nargs="+", default=DISEASES)
    parser.add_argument("--seed-eval", action="store_true",
                        help="Skip the study; retrain its best config over --seeds seeds")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Epoch cap. The default relies on early stopping; lower it to rehearse.")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick sanity run: 1 dataset, 1 trial, few epochs")
    args = parser.parse_args()

    dataset = resolve_dataset(args.dataset)
    resolve_node_input(args.node_input)
    if args.seed_eval:
        run_seed_eval(dataset, args.graph, args.backbone, args.seeds, args.k, args.diseases,
                      epochs=args.epochs)
    elif args.smoke:
        run_study(dataset, args.graph, args.backbone, trials=1, k_range=args.k,
                  quantiles=args.quantiles, k_cores=args.k_cores, sim_spaces=args.sim_spaces,
                  diseases=args.diseases[:1], epochs=15)
    else:
        run_study(dataset, args.graph, args.backbone, trials=args.trials, k_range=args.k,
                  quantiles=args.quantiles, k_cores=args.k_cores, sim_spaces=args.sim_spaces,
                  diseases=args.diseases, epochs=args.epochs)


if __name__ == "__main__":
    main()
