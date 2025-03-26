import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

from main import load_edge_data
from models import TrajectoryModel

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
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    X, Y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        path = row['path']
        if len(path) < 2:
            continue
        if len(path) > max_len:
            path = path[:max_len]
        else:
            path += [padding_id] * (max_len - len(path))
        X.append(path)
        Y.append(row['total_time'])

    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.float32)


def evaluation(model, data_path, file_list, padding_id, max_len=64):
    print("\n=== Travel Time Estimation ===")
    model.eval()
    X, Y = load_task_data(data_path, file_list, padding_id, max_len)
    X, Y = X.cuda(), Y.cuda()

    # Shuffle
    indices = torch.randperm(X.size(0))
    X, Y = X[indices], Y[indices]

    split = int(0.2 * X.size(0))
    X_train, Y_train = X[split:], Y[split:]
    X_val, Y_val = X[:split], Y[:split]

    # === Get trajectory embeddings ===
    with torch.no_grad():
        X_train_embed = model.encode_sequence(X_train)
        X_val_embed = model.encode_sequence(X_val)

    reg = MLPRegressor(input_dim=X_train_embed.shape[1]).cuda()
    opt = torch.optim.Adam(reg.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_mae, best_rmse = float('inf'), float('inf')
    for epoch in range(1, 51):
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


from models import TrajectoryModel
from main import load_edge_data

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
for filename in os.listdir(checkpoint_dir):
    if re.match(r"model_epoch_\d+\.pt", filename):
        checkpoint_files.append(os.path.join(checkpoint_dir, filename))

if not checkpoint_files:
    print(f"No model_epoch_*.pt files found in {checkpoint_dir}")
    exit()

checkpoint_files.sort(key=lambda f: int(re.search(r"model_epoch_(\d+)\.pt", f).group(1)), reverse=True)

for checkpoint_file in checkpoint_files:
    try:
        print(f"Loading checkpoint: {checkpoint_file}")
        model = TrajectoryModel(config, graph_data)
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.cuda().eval()
        evaluation(model, data_path, task_files, padding_id=config['num_nodes'])
    except Exception as e:
        print(f"Error loading or evaluating {checkpoint_file}: {e}")
