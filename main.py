import os
import pandas as pd
import ast
from tqdm import tqdm

from TrajectoryDataset import LazyTrajectoryDataset, TrajectoryDataset
from models import TrajectoryModel

import torch

def train(model, dataloader, graph_data, spatial_grid, optimizer, config, device, epoch=None):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)

    x, edge_index = graph_data
    x, edge_index = x.to(device), edge_index.to(device)
    spatial_grid = spatial_grid.to(device)

    pbar = tqdm(dataloader, desc=f"[Epoch {epoch}] Training", leave=True)
    for batch_idx, batch in enumerate(pbar):
        # Move tensors to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward
        L_mtm, L_mtd, L_hour, L_weekday = model(batch, spatial_grid)
        loss = (config['λ1'] * L_mtm +
                config['λ2'] * L_mtd +
                config['λ3'] * L_hour +
                config['λ4'] * L_weekday)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Update tqdm bar with batch info
        pbar.set_postfix({
            "Batch Loss": f"{loss.item():.4f}",
            "MTM": f"{L_mtm.item():.4f}",
            "MTD": f"{L_mtd.item():.4f}",
            "HOUR": f"{L_hour.item():.4f}",
            "WEKDAY": f"{L_weekday.item():.4f}",
            "Avg Loss": f"{total_loss / (batch_idx + 1):.4f}"
        })

    return total_loss / num_batches

@torch.no_grad()
def validate(model, dataloader, graph_data, spatial_grid, config, device):
    model.eval()
    total_loss = 0
    total_mtm = 0
    total_mtd = 0
    total_hour = 0
    total_weekday = 0
    num_batches = len(dataloader)

    x, edge_index = graph_data
    x, edge_index = x.to(device), edge_index.to(device)
    spatial_grid = spatial_grid.to(device)

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward
        L_mtm, L_mtd, L_hour, L_weekday = model(batch, spatial_grid)
        loss = (config['λ1'] * L_mtm +
                config['λ2'] * L_mtd +
                config['λ3'] * L_hour +
                config['λ4'] * L_weekday)

        total_loss += loss.item()
        total_mtm += L_mtm.item()
        total_mtd += L_mtd.item()
        total_hour += L_hour.item()
        total_weekday += L_weekday.item()

    return {
        'val_loss': total_loss / num_batches,
        'val_mtm': total_mtm / num_batches,
        'val_mtd': total_mtd / num_batches,
        'val_hour': total_hour / num_batches,
        'val_weekday': total_weekday / num_batches,
    }

# utils.py
def load_all_trajectories(data_path, use_files=None):
    trajectories = []
    for fname in sorted(os.listdir(data_path)):
        if fname.endswith(".csv") and fname != "edge_features.csv":
            if use_files and fname not in use_files:
                continue
            print("Loading:", fname)
            df = pd.read_csv(os.path.join(data_path, fname))
            for _, row in tqdm(df.iterrows(), total=df.shape[0]):
                try:
                    traj = {
                        'order_id': row['order_id'],
                        'path': ast.literal_eval(row['path']),
                        'timestamps': ast.literal_eval(row['timestamp']),
                        'pass_time': ast.literal_eval(row['pass_time']),
                        'total_time': row['total_time']
                    }
                    trajectories.append(traj)
                except:
                    continue
    return trajectories

import os
import ast
import pandas as pd

def optimized_load_trajectory_data(data_path, include_files=None):
    """
    Optimized version of loading trajectory data using preloaded and concatenated DataFrames.
    """
    all_files = sorted([
        os.path.join(data_path, f)
        for f in os.listdir(data_path)
        if f.endswith('.csv') and f != 'edge_features.csv'
    ])

    if include_files is not None:
        all_files = [os.path.join(data_path, f) for f in include_files]

    # Preload and concatenate all data
    dfs = []
    for file_path in all_files:
        print("Loading file: ", file_path)
        df = pd.read_csv(file_path, converters={
            'path': ast.literal_eval,
            'timestamp': ast.literal_eval,
            'pass_time': ast.literal_eval
        })
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    trajectories = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        order_id = row['order_id']
        path = row['path']
        timestamps = row['timestamp']
        pass_time = row['pass_time']
        total_time = row['total_time']

        if len(timestamps) != len(path) + 1 or len(pass_time) != len(path):
            continue

        trajectories.append({
            'order_id': order_id,
            'path': path,
            'timestamps': timestamps,
            'pass_time': pass_time,
            'total_time': total_time
        })

    return trajectories
    


import numpy as np

def load_edge_data(edge_features_path, edge_index_path):
    edge_features_df = pd.read_csv(edge_features_path)
    edge_index = np.load(edge_index_path)  # shape: [2, num_edges]

    return edge_features_df, edge_index

from tqdm import tqdm

import os
import torch
from torch.utils.data import DataLoader
if __name__ == "__main__":
    # === Your existing paths ===
    data_path = "datasets/didi_chengdu"
    edge_features_path = os.path.join(data_path, "edge_features.csv")
    edge_index_path = os.path.join(data_path, "line_graph_edge_idx.npy")
    dicts_pkl_path = os.path.join(data_path, "dicts.pkl")

    # === Step 1: Load trajectory and graph data ===
    edge_features_df, edge_index = load_edge_data(edge_features_path, edge_index_path)
    all_csvs = sorted([f for f in os.listdir(data_path) if f.endswith(".csv") and f != "edge_features.csv"])
    train_csvs, val_csvs = all_csvs[:-1], all_csvs[-1:]

    train_trajs = optimized_load_trajectory_data(data_path, include_files=train_csvs)
    val_trajs = optimized_load_trajectory_data(data_path, include_files=val_csvs)


    train_dataset = TrajectoryDataset(train_trajs, edge_features_df)
    val_dataset = TrajectoryDataset(val_trajs, edge_features_df)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=val_dataset.collate_fn)

    # === Step 2: Load node features for GAT ===
    node_features = torch.tensor(edge_features_df[['oneway', 'lanes', 'highway_id', 'length_id',
                                                'bridge', 'tunnel', 'road_speed', 'traj_speed']].values,
                                dtype=torch.float32)  # [num_nodes, feat_dim]

    # === Step 3: Generate Spatial Grid ===
    from GridGenerator import generate_spatial_grid  # assuming you saved it in spatial_grid.py

    map_bbox = [30.730, 30.6554, 104.127, 104.0397]  # Chengdu bounding box
    graph_pkl_path = os.path.join('datasets/didi_chengdu/osm_graph', 'ChengDu.pkl')  # or wherever your graph is cached
    spatial_grid_np = generate_spatial_grid(
        edge_features_path,
        dicts_pkl_path,
        graph_pkl_path,
        map_bbox=[30.730, 30.6554, 104.127, 104.0397],
        grid_size=(64, 64)
    )
    spatial_grid_tensor = torch.tensor(spatial_grid_np, dtype=torch.float32).unsqueeze(0)  # [1, C, H, W]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graph_data = (node_features, torch.tensor(edge_index, dtype=torch.long))
    spatial_grid_tensor = spatial_grid_tensor.to(device)

    # === Step 5: Config and Model ===
    config = {
        'node_feat_dim': node_features.shape[1],
        'gat_hidden': 64,
        'embed_dim': 128,
        'spatial_in_channels': spatial_grid_tensor.shape[1],
        'num_heads': 4,
        'spatial_layers': 2,
        'traj_layers': 2,
        'num_nodes': node_features.shape[0],
        'λ1': 1.0,  # MTM
        'λ2': 1.0,  # MTD
        'λ3': 1.0,  # Hour
        'λ4': 1.0,  # Weekday
    }

    model = TrajectoryModel(config, graph_data)
    # device = "cpu"
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # === Step 6: Call training ===

    for epoch in range(1, 11):
        avg_loss = train(model, train_loader, graph_data, spatial_grid_tensor, optimizer, config, device, epoch)
        val_metrics = validate(model, val_loader, graph_data, spatial_grid_tensor, config, device)

        print(f"[Epoch {epoch}] | Train Loss: {avg_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f}\n" 
            f"(MTM: {val_metrics['val_mtm']:.4f}, MTD: {val_metrics['val_mtd']:.4f}, HOUR: {val_metrics['val_hour']:.4f}, WEEKDAY: {val_metrics['val_weekday']:.4f})\n"
            f"(MTM: {val_metrics['val_mtm']:.4f}, MTD: {val_metrics['val_mtd']:.4f}, HOUR: {val_metrics['val_hour']:.4f}, WEEKDAY: {val_metrics['val_weekday']:.4f})\n")

        # Save model
        save_path = os.path.join("checkpoints", f"model_epoch_{epoch}.pt")
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_loss,
            'val_metrics': val_metrics
        }, save_path)









