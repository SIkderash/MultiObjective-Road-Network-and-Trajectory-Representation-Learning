import os
import pandas as pd
import ast
from tqdm import tqdm

from TrajectoryDataset import TrajectoryDataset
from models import TrajectoryModel

import torch
import numpy as np
from torch.utils.data import DataLoader
from GridGenerator import generate_spatial_grid

def train(model, dataloader, graph_data, spatial_grid, optimizer, config, device, epoch=None):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)

    x, edge_index = graph_data
    x, edge_index = x.to(device), edge_index.to(device)
    spatial_grid = spatial_grid.to(device)

    pbar = tqdm(dataloader, desc=f"[Epoch {epoch}] Training", leave=True)
    for batch_idx, batch in enumerate(pbar):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        L_mtm, L_contrastive = model(batch, spatial_grid)
        loss = config['λ1'] * L_mtm + config['λ2'] * L_contrastive

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pbar.set_postfix({
            "Batch Loss": f"{loss.item():.4f}",
            "MTM": f"{L_mtm.item():.4f}",
            "CL": f"{L_contrastive.item():.4f}",
            "Avg Loss": f"{total_loss / (batch_idx + 1):.4f}"
        })

    return total_loss / num_batches



@torch.no_grad()
def validate(model, dataloader, graph_data, spatial_grid, config, device):
    model.eval()
    total_loss = 0
    total_mtm = 0
    total_cl = 0
    num_batches = len(dataloader)

    x, edge_index = graph_data
    x, edge_index = x.to(device), edge_index.to(device)
    spatial_grid = spatial_grid.to(device)

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        L_mtm, L_contrastive = model(batch, spatial_grid)

        loss = config['λ1'] * L_mtm + config['λ2'] * L_contrastive
        total_loss += loss.item()
        total_mtm += L_mtm.item()
        total_cl += L_contrastive.item()

    return {
        'val_loss': total_loss / num_batches,
        'val_mtm': total_mtm / num_batches,
        'val_cl': total_cl / num_batches,
    }


def optimized_load_trajectory_data(data_path, include_files=None):
    all_files = sorted([
        os.path.join(data_path, f)
        for f in os.listdir(data_path)
        if f.endswith('.csv') and f != 'edge_features.csv'
    ])

    if include_files is not None:
        all_files = [os.path.join(data_path, f) for f in include_files]

    dfs = []


    use_one_file = True
    for file_path in all_files:
        print("Loading file: ", file_path)
        df = pd.read_csv(file_path, converters={
            'path': ast.literal_eval,
            'timestamp': ast.literal_eval,
            'pass_time': ast.literal_eval
        })
        dfs.append(df)
        if use_one_file:
            break

    df = pd.concat(dfs, ignore_index=True)

    trajectories = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        path = row['path']
        timestamps = row['timestamp']
        pass_time = row['pass_time']

        if len(timestamps) != len(path) + 1 or len(pass_time) != len(path):
            continue

        trajectories.append({
            'order_id': row['order_id'],
            'path': path,
            'timestamps': timestamps,
            'pass_time': pass_time,
            'total_time': row['total_time']
        })

    return trajectories

def load_edge_data(edge_features_path, edge_index_path):
    edge_features_df = pd.read_csv(edge_features_path)
    edge_index = np.load(edge_index_path)
    return edge_features_df, edge_index

if __name__ == "__main__":
    data_path = "datasets/didi_chengdu"
    edge_features_path = os.path.join(data_path, "edge_features.csv")
    edge_index_path = os.path.join(data_path, "line_graph_edge_idx.npy")
    dicts_pkl_path = os.path.join(data_path, "dicts.pkl")

    edge_features_df, edge_index = load_edge_data(edge_features_path, edge_index_path)
    all_csvs = sorted([f for f in os.listdir(data_path) if f.endswith(".csv") and f != "edge_features.csv"])
    train_csvs, val_csvs = all_csvs[:-1], all_csvs[-1:]

    train_trajs = optimized_load_trajectory_data(data_path, include_files=train_csvs)
    val_trajs = optimized_load_trajectory_data(data_path, include_files=val_csvs)

    train_dataset = TrajectoryDataset(train_trajs, edge_features_df)
    val_dataset = TrajectoryDataset(val_trajs, edge_features_df)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=val_dataset.collate_fn)

    node_features = torch.tensor(edge_features_df[[
        'oneway', 'lanes', 'highway_id', 'length_id',
        'bridge', 'tunnel', 'road_speed', 'traj_speed']].values,
        dtype=torch.float32)

    map_bbox = [30.730, 30.6554, 104.127, 104.0397]
    graph_pkl_path = os.path.join('datasets/didi_chengdu/osm_graph', 'ChengDu.pkl')
    spatial_grid_np = generate_spatial_grid(
        edge_features_path,
        dicts_pkl_path,
        graph_pkl_path,
        map_bbox=map_bbox,
        grid_size=(64, 64)
    )
    spatial_grid_tensor = torch.tensor(spatial_grid_np, dtype=torch.float32).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    graph_data = (node_features, torch.tensor(edge_index, dtype=torch.long))
    spatial_grid_tensor = spatial_grid_tensor.to(device)

    config = {
        'node_feat_dim': node_features.shape[1],
        'gat_hidden': 64,
        'embed_dim': 128,
        'spatial_in_channels': spatial_grid_tensor.shape[1],
        'num_heads': 4,
        'spatial_layers': 2,
        'traj_layers': 2,
        'num_nodes': node_features.shape[0],
        'λ1': 1.0,
        'λ2': 5.0
    }

    model = TrajectoryModel(config, graph_data).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 5):
        avg_loss = train(model, train_loader, graph_data, spatial_grid_tensor, optimizer, config, device, epoch)
        val_metrics = validate(model, val_loader, graph_data, spatial_grid_tensor, config, device)

        print(f"[Epoch {epoch}] | Train Loss: {avg_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f} | MTM: {val_metrics['val_mtm']:.4f} | CL: {val_metrics['val_cl']:.4f}")

        save_path = os.path.join("checkpoints", f"model_epoch_{epoch}.pt")
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_loss,
            'val_metrics': val_metrics
        }, save_path)