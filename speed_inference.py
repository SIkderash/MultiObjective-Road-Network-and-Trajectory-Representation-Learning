import os
import re
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

from models import TrajectoryModel
from main import load_edge_data

# Reproducibility
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Regressor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x).squeeze(1)

def parse_config_from_filename(filename):
    config = {
        'use_temporal_encoding': False,
        'use_time_embeddings': False,
        'use_spatial_fusion': False,
        'use_traj_traj_cl': False,
        'use_traj_node_cl': False,
        'use_node_node_cl': False,
        'contrastive_type': 'infonce'
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

def evaluation(model, feature_df, fold=5):
    print("\n--- Speed Inference (Regression) ---")
    model.eval()
    with torch.no_grad():
        x = model.encode_graph().detach().cpu()

    y = torch.tensor(feature_df['road_speed'].values, dtype=torch.float32)
    node_ids = feature_df['road_id'].values
    x = x[node_ids]

    kf = KFold(n_splits=fold, shuffle=True, random_state=42)
    y_preds, y_trues = [], []

    for train_idx, val_idx in kf.split(x):
        x_train, y_train = x[train_idx], y[train_idx]
        x_val, y_val = x[val_idx], y[val_idx]

        regressor = Regressor(x.shape[1]).to(device)
        opt = torch.optim.Adam(regressor.parameters(), lr=1e-3)

        best_pred = None
        best_mse = float('inf')

        for _ in range(100):
            regressor.train()
            opt.zero_grad()
            pred = regressor(x_train.to(device))
            loss = nn.MSELoss()(pred, y_train.to(device))
            loss.backward()
            opt.step()

            regressor.eval()
            y_pred = regressor(x_val.to(device)).detach().cpu()
            mse = mean_squared_error(y_val, y_pred)
            if mse < best_mse:
                best_mse = mse
                best_pred = y_pred

        y_preds.append(best_pred)
        y_trues.append(y_val)

    y_preds = torch.cat(y_preds)
    y_trues = torch.cat(y_trues)

    mae = mean_absolute_error(y_trues, y_preds)
    rmse = mean_squared_error(y_trues, y_preds, squared=False)
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")

if __name__ == "__main__":
    print("\n=== Speed Inference Evaluation ===")

    data_path = "datasets/didi_chengdu"
    edge_features_path = os.path.join(data_path, "edge_features.csv")
    edge_index_path = os.path.join(data_path, "line_graph_edge_idx.npy")

    edge_features_df, edge_index = load_edge_data(edge_features_path, edge_index_path)
    node_features = torch.tensor(edge_features_df[[ 
        'oneway', 'lanes', 'highway_id', 'length_id',
        'bridge', 'tunnel', 'road_speed', 'traj_speed'
    ]].values, dtype=torch.float32)

    graph_data = (node_features, torch.tensor(edge_index, dtype=torch.long))

    checkpoint_dir = "checkpoints"
    ablation_tag = ""  # change as needed
    checkpoint_files = [
        os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir)
        if f.endswith(".pt") and ablation_tag in f
    ]
    checkpoint_files.sort(key=lambda f: int(re.search(r"model_epoch_(\d+)", f).group(1)))

    if not checkpoint_files:
        print(f"No matching checkpoint files found for tag '{ablation_tag}' in {checkpoint_dir}")
        exit()

    for checkpoint_file in checkpoint_files:
        try:
            print(f"\nLoading checkpoint: {checkpoint_file}")
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
            config.update(parse_config_from_filename(checkpoint_file))

            model = TrajectoryModel(config, graph_data).to(device)
            checkpoint = torch.load(checkpoint_file, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            evaluation(model, edge_features_df, fold=5)

        except Exception as e:
            print(f"Error loading or evaluating {checkpoint_file}: {e}")
