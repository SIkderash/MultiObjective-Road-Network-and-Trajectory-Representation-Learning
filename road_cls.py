import os
import re
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from GridGenerator import generate_spatial_grid
from main import load_edge_data
from models import TrajectoryModel
import random
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Model Definition ===
class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# === Metric Calculation ===
def compute_metrics(y_true, y_pred):
    return {
        "Micro F1": f1_score(y_true, y_pred, average="micro"),
        "Macro F1": f1_score(y_true, y_pred, average="macro"),
        "Accuracy": accuracy_score(y_true, y_pred),
    }

def set_seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# === Evaluation Function ===
def evaluation(model, feature_df, spatial_grid_tensor=None, fold=5):
    print("\n--- Road Classification ---")
    model.eval()

    with torch.no_grad():
        if model.config.get('use_spatial_fusion', False) and spatial_grid_tensor is not None:
            x, _, _ = model.encode_graph(spatial_grid_tensor.to(device))
        else:
            x, _, _ = model.encode_graph()

    if isinstance(x, tuple): x = x[0]
    x = x.cpu()

    valid_labels = ['primary', 'secondary', 'tertiary', 'residential']
    label_to_id = {lbl: i for i, lbl in enumerate(valid_labels)}
    y_df = feature_df[feature_df['highway'].isin(valid_labels)].copy()
    y_df = y_df.sort_values('road_id')

    x = x[y_df['road_id'].values]
    y = torch.tensor(y_df['highway'].map(label_to_id).values)

    skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42)
    y_preds, y_trues = [], []

    for train_idx, test_idx in skf.split(x, y):
        set_seed_all()
        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]

        clf = Classifier(x.shape[1], len(valid_labels)).to(device)
        opt = torch.optim.Adam(clf.parameters(),
                               lr=model.config.get("learning_rate", 1e-2),
                               weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='max', factor=model.config.get("scheduler_factor", 0.8),
            patience=model.config.get("scheduler_patience", 10),
            threshold=1e-3, min_lr=1e-7
        )

        loss_fn = nn.CrossEntropyLoss() if model.config.get("loss_fn", "ce") == "ce" else nn.NLLLoss()
        best_acc, patience_counter = 0, 0
        best_pred = None
        early_stop_patience = model.config.get("early_stop_patience", 30)

        for epoch in range(1, 10000):
            clf.train()
            opt.zero_grad()
            loss = loss_fn(clf(x_train.to(device)), y_train.to(device))
            loss.backward()
            opt.step()

            clf.eval()
            preds = clf(x_test.to(device)).argmax(dim=1).cpu()
            acc = accuracy_score(y_test, preds)
            scheduler.step(acc)

            # if epoch % 100 == 0:
                # print(f"[Epoch {epoch}] LR: {opt.param_groups[0]['lr']:.6f}, Acc: {acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_pred = preds
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}. Best acc: {best_acc:.4f}")
                break

        y_preds.append(best_pred)
        y_trues.append(y_test)

    y_preds = torch.cat(y_preds)
    y_trues = torch.cat(y_trues)
    metrics = compute_metrics(y_trues, y_preds)
    print(f'Micro F1: {metrics["Micro F1"]:.4f}')
    print(f'Macro F1: {metrics["Macro F1"]:.4f}')


# === Filename-Based Config Parser ===
def parse_config_from_filename(filename):
    config = {
        'use_node_embedding': False,
        'use_temporal_encoding': False,
        'use_time_embeddings': False,
        'use_spatial_fusion': False,
        'use_traj_traj_cl': False,
        'use_traj_node_cl': False,
        'use_node_node_cl': False,
        'contrastive_type': 'infonce',
    }

    filename = filename.lower()

    if "node_embed" in filename:
        config['use_node_embedding'] = True
        print("Using Node Embedding")
    if "temporal" in filename:
        config['use_temporal_encoding'] = True
        print("Using Temporal Encoding")
    if "timeemb" in filename:
        config['use_time_embeddings'] = True
        print("Using Time Embeddings")
    if "spatialfusion" in filename:
        config['use_spatial_fusion'] = True
        print("Using Spatial Fusion")
    if "trajtraj" in filename:
        config['use_traj_traj_cl'] = True
        print("Using Traj-Traj CL")
    if "trajnode" in filename:
        config['use_traj_node_cl'] = True
        print("Using Traj-Node CL")
    if "nodenode" in filename:
        config['use_node_node_cl'] = True
        print("Using Node-Node CL")
    if "jsd" in filename:
        config['contrastive_type'] = "jsd"
        print("Using JSD")
    return config


# === Entry Point ===
if __name__ == "__main__":
    print("\n=== Road Classification Evaluation ===")
    set_seed_all()

    data_path = "datasets/didi_chengdu"
    edge_features_path = os.path.join(data_path, "edge_features.csv")
    edge_index_path = os.path.join(data_path, "line_graph_edge_idx.npy")

    edge_features_df, edge_index = load_edge_data(edge_features_path, edge_index_path)
    node_features = torch.tensor(edge_features_df[['oneway', 'lanes', 'highway_id', 'length_id',
                                                    'bridge', 'tunnel', 'road_speed', 'traj_speed']].values,
                                 dtype=torch.float32)
    graph_data = (node_features, torch.tensor(edge_index, dtype=torch.long))

    dicts_pkl_path = os.path.join(data_path, "dicts.pkl")
    graph_pkl_path = os.path.join(data_path, "osm_graph", "ChengDu.pkl")
    map_bbox = [30.730, 30.6554, 104.127, 104.0397]

    spatial_grid_np = generate_spatial_grid(edge_features_path, dicts_pkl_path, graph_pkl_path,
                                            map_bbox=map_bbox, grid_size=(64, 64))
    spatial_grid_tensor = torch.tensor(spatial_grid_np, dtype=torch.float32).unsqueeze(0)

    checkpoint_dir = "checkpoints"
    # checkpoint_dir = "Models/MTM"
    checkpoint_files = [os.path.join(checkpoint_dir, f)
                        for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]

    def extract_epoch(filename):
        match = re.search(r"epoch[_\-]?(\d+)", filename)
        return int(match.group(1)) if match else float('inf')

    checkpoint_files.sort(key=lambda f: extract_epoch(f))

    if not checkpoint_files:
        print("❌ No model_epoch_*.pt files found.")
        exit()

    for ckpt_path in checkpoint_files:
        try:
            set_seed_all()
            print(f"\n📦 Loading checkpoint: {ckpt_path}")
            config_base = {
                'node_feat_dim': node_features.shape[1],
                'gat_hidden': 64,
                'embed_dim': 128,
                'spatial_in_channels': 8,
                'num_heads': 4,
                'spatial_layers': 2,
                'traj_layers': 2,
                'num_nodes': node_features.shape[0],

                # Task-specific hyperparameters
                'learning_rate': 1e-2,
                'scheduler_factor': 0.8,
                'scheduler_patience': 10,
                'loss_fn': 'ce',
                'early_stop_patience': 200,
            }
            config = {**config_base, **parse_config_from_filename(ckpt_path)}

            model = TrajectoryModel(config, graph_data).to(device)
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            evaluation(model, edge_features_df, spatial_grid_tensor if config['use_spatial_fusion'] else None)

        except Exception as e:
            print(f"❌ Error loading or evaluating {ckpt_path}: {e}")
