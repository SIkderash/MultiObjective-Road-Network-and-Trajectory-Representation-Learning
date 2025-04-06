from datetime import datetime
import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

from GridGenerator import generate_spatial_grid
from main import load_edge_data
from models import TrajectoryModel

# Reproducibility
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


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


class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x).squeeze(-1)


def load_task_data(data_path, file_list, padding_id, max_len=64):
    dfs = []
    for fname in file_list:
        df = pd.read_csv(os.path.join(data_path, fname))
        df['path'] = df['path'].map(eval)
        df['timestamp'] = df['timestamp'].map(eval)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    X_dict, Y = [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        path = row['path']
        timestamps = row['timestamp']

        if len(path) < 2 or len(timestamps) < 1:
            continue

        if len(timestamps) == len(path) + 1:
            timestamps = timestamps[:-1]

        path = path[:max_len]
        timestamps = timestamps[:max_len]

        pad_len = max_len - len(path)
        path += [padding_id] * pad_len
        timestamps += [timestamps[-1]] * pad_len if timestamps else [0] * pad_len

        hours = [datetime.utcfromtimestamp(ts).hour for ts in timestamps]
        weekdays = [datetime.utcfromtimestamp(ts).weekday() for ts in timestamps]

        X_dict.append({
            'road_nodes': path,
            'timestamps': timestamps,
            'hour': hours,
            'weekday': weekdays
        })
        Y.append(row['total_time'])

    X = {
        'road_nodes': torch.tensor([x['road_nodes'] for x in X_dict], dtype=torch.long),
        'timestamps': torch.tensor([x['timestamps'] for x in X_dict], dtype=torch.float32),
        'hour': torch.tensor([x['hour'] for x in X_dict], dtype=torch.long),
        'weekday': torch.tensor([x['weekday'] for x in X_dict], dtype=torch.long)
    }

    Y = torch.tensor(Y, dtype=torch.float32)
    return X, Y

def batched_encode(model, X_dict, batch_size=256, spatial_grid=None):
    all_embeds = []
    for i in range(0, len(X_dict['road_nodes']), batch_size):
        batch = {k: v[i:i+batch_size].to(device) for k, v in X_dict.items()}
        embeds = model.encode_sequence(
            batch['road_nodes'], batch['timestamps'], batch['hour'], batch['weekday'],
            spatial_grid=spatial_grid
        )
        all_embeds.append(embeds.cpu())
    return torch.cat(all_embeds, dim=0)

def evaluation(model, data_path, file_list, actual_spatial_grid_tensor, padding_id, max_len=64):
    print("\n=== Travel Time Estimation ===")
    model.eval()

    X_dict, Y = load_task_data(data_path, file_list, padding_id, max_len)

    # Convert to device
    X_dict = {k: v.to(device) for k, v in X_dict.items()}
    Y = Y.to(device)

    # Shuffle
    indices = torch.randperm(Y.size(0))
    X_dict = {k: v[indices] for k, v in X_dict.items()}
    Y = Y[indices]

    # Split
    split = int(0.2 * Y.size(0))
    X_train = {k: v[split:] for k, v in X_dict.items()}
    X_val   = {k: v[:split] for k, v in X_dict.items()}
    Y_train, Y_val = Y[split:], Y[:split]

    try:
        with torch.no_grad():
                spatial_grid = actual_spatial_grid_tensor.to(device) if config['use_spatial_fusion'] else None
                X_train_embed = batched_encode(model, X_train, spatial_grid=spatial_grid)
                X_val_embed = batched_encode(model, X_val, spatial_grid=spatial_grid)
    except Exception as e:
        print("Caught during encode_sequence:", e)
        raise

    reg = MLPRegressor(input_dim=X_train_embed.shape[1]).to(device)
    opt = torch.optim.Adam(reg.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    best_mae, best_rmse = float('inf'), float('inf')
    for epoch in range(1, 1001):
        reg.train()
        opt.zero_grad()
        pred = reg(X_train_embed)
        loss = criterion(pred, Y_train)
        loss.backward()
        opt.step()

        reg.eval()
        with torch.no_grad():
            pred_val = reg(X_val_embed)
            mae = mean_absolute_error(Y_val.cpu(), pred_val.cpu())
            rmse = mean_squared_error(Y_val.cpu(), pred_val.cpu()) ** 0.5

        print(f"[Epoch {epoch}] MAE: {mae:.4f} | RMSE: {rmse:.4f}")
        best_mae = min(best_mae, mae)
        best_rmse = min(best_rmse, rmse)

    print(f"\n>>> Best MAE: {best_mae:.4f}, Best RMSE: {best_rmse:.4f}")


# === Entry ===
if __name__ == "__main__":
    data_path = "datasets/didi_chengdu"
    all_csvs = sorted([f for f in os.listdir(data_path) if f.endswith(".csv") and f != "edge_features.csv"])
    task_files = all_csvs[-1:]
    edge_features_path = os.path.join(data_path, "edge_features.csv")
    edge_index_path = os.path.join(data_path, "line_graph_edge_idx.npy")

    edge_features_df, edge_index = load_edge_data(edge_features_path, edge_index_path)
    node_features = torch.tensor(edge_features_df[[ 
        'oneway', 'lanes', 'highway_id', 'length_id',
        'bridge', 'tunnel', 'road_speed', 'traj_speed'
    ]].values, dtype=torch.float32)

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
    
    checkpoint_dir = "checkpoints"
    ablation_tag = ""  # ← update tag for filtering if needed
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

            evaluation(model, data_path, task_files, spatial_grid_tensor, padding_id=config['num_nodes'])

        except Exception as e:
            print(f"Error loading or evaluating {checkpoint_file}: {e}")