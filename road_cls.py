import os
import re
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from GridGenerator import generate_spatial_grid
from main import load_edge_data
from models import TrajectoryModel

import os
import random
import numpy as np
import torch

# Set seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# For deterministic behavior (slower but reproducible)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)

def evaluation(model, feature_df, fold=5):
    print("\n--- Road Classification ---")
    model.eval()
    x = model.encode_graph()  # [num_nodes, embed_dim]
    # x = model.encode_graph_with_context(spatial_grid_tensor)
    if isinstance(x, tuple):
        x = x[0]
    x = x.detach().cpu()
    # print(feature_df.columns)
    valid_labels = ['primary', 'secondary', 'tertiary', 'residential']
    id_dict = {lbl: i for i, lbl in enumerate(valid_labels)}
    y_df = feature_df[feature_df['highway'].isin(valid_labels)].copy()
    y_df = y_df.sort_values('road_id')

    node_indices = y_df['road_id'].values
    x = x[node_indices]
    y = torch.tensor(y_df['highway'].map(id_dict).values)

    # === Stratified K-Fold ===
    skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42)

    y_preds, y_trues = [], []
    for train_idx, test_idx in skf.split(x, y):
        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]

        clf = Classifier(x.shape[1], len(valid_labels)).cuda()
        opt = torch.optim.Adam(clf.parameters(), lr=1e-2)

        best_pred = None
        best_acc = 0
        for epoch in range(1, 501):
            clf.train()
            opt.zero_grad()
            loss = nn.CrossEntropyLoss()(clf(x_train.cuda()), y_train.cuda())
            loss.backward()
            opt.step()

            clf.eval()
            preds = clf(x_test.cuda()).argmax(dim=1).cpu()
            acc = accuracy_score(y_test, preds)
            if acc > best_acc:
                best_acc = acc
                best_pred = preds

        y_preds.append(best_pred)
        y_trues.append(y_test)

    y_preds = torch.cat(y_preds)
    y_trues = torch.cat(y_trues)
    print(f'Micro F1: {f1_score(y_trues, y_preds, average="micro"):.4f}')
    print(f'Macro F1: {f1_score(y_trues, y_preds, average="macro"):.4f}')


def parse_config_from_filename(filename):
    config = {
        'use_temporal_encoding': False,
        'use_time_embeddings': False,
        'use_spatial_fusion': False,
        'use_traj_traj_cl': False,
        'use_traj_node_cl': False,
        'use_node_node_cl': False,
        'contrastive_type': 'infonce',  # default
    }

    filename = filename.lower()

    if "temporal" in filename:
        config['use_temporal_encoding'] = True
    if "timeemb" in filename:
        config['use_time_embeddings'] = True
    if "spatialfusion" in filename:
        config['use_spatial_fusion'] = True
    if "trajtraj" in filename:
        config['use_traj_traj_cl'] = True
    if "trajnode" in filename:
        config['use_traj_node_cl'] = True
    if "nodenode" in filename:
        config['use_node_node_cl'] = True
    if "jsd" in filename:
        config['contrastive_type'] = "jsd"

    return config


# === Main Entry ===
if __name__ == "__main__":
    print("\n=== Road Classification Evaluation ===")

    data_path = "datasets/didi_chengdu"
    edge_features_path = os.path.join(data_path, "edge_features.csv")
    edge_index_path = os.path.join(data_path, "line_graph_edge_idx.npy")

    edge_features_df, edge_index = load_edge_data(edge_features_path, edge_index_path)
    node_features = torch.tensor(edge_features_df[['oneway', 'lanes', 'highway_id', 'length_id',
                                                    'bridge', 'tunnel', 'road_speed', 'traj_speed']].values,
                                 dtype=torch.float32)
    graph_data = (node_features, torch.tensor(edge_index, dtype=torch.long))

    map_bbox = [30.730, 30.6554, 104.127, 104.0397]
    dicts_pkl_path = os.path.join(data_path, "dicts.pkl")
    graph_pkl_path = os.path.join('datasets/didi_chengdu/osm_graph', 'ChengDu.pkl')
    spatial_grid_np = generate_spatial_grid(
        edge_features_path,
        dicts_pkl_path,
        graph_pkl_path,
        map_bbox=map_bbox,
        grid_size=(64, 64)
    )
    spatial_grid_tensor = torch.tensor(spatial_grid_np, dtype=torch.float32).unsqueeze(0)

    config = {
        'node_feat_dim': node_features.shape[1],
        'gat_hidden': 64,
        'embed_dim': 128,
        'spatial_in_channels': 8,
        'num_heads': 4,
        'spatial_layers': 2,
        'traj_layers': 2,
        'num_nodes': node_features.shape[0],
    }


    checkpoint_dir="checkpoints"
    checkpoint_files = []

    ablation_tag = ""  # can be 'full', 'baseline', etc.

    checkpoint_files = [
        os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir)
        if f.endswith(".pt") and ablation_tag in f
    ]

    # Sort by epoch number if available in filename
    def extract_epoch(filename):
        match = re.search(r"epoch[_\-]?(\d+)", filename)
        return int(match.group(1)) if match else float('inf')

    checkpoint_files.sort(key=lambda f: extract_epoch(f))

    if not checkpoint_files:
        print(f"No model_epoch_*.pt files found in {checkpoint_dir}")
        exit()

    for checkpoint_file in checkpoint_files:
        try:
            print(f"Loading checkpoint: {checkpoint_file}")

            # === Base Config ===
            config = {
                'node_feat_dim': node_features.shape[1],
                'gat_hidden': 64,
                'embed_dim': 128,
                'spatial_in_channels': 8,
                'num_heads': 4,
                'spatial_layers': 2,
                'traj_layers': 2,
                'num_nodes': node_features.shape[0],
            }

            # === Augment config from filename ===
            config.update(parse_config_from_filename(checkpoint_file))

            # === Load model and checkpoint ===
            model = TrajectoryModel(config, graph_data).cuda()
            checkpoint = torch.load(checkpoint_file)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.eval()

            evaluation(model, edge_features_df, fold=5)

        except Exception as e:
            print(f"Error loading or evaluating {checkpoint_file}: {e}")
