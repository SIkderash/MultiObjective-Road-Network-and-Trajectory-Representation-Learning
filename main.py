import os
import re
import argparse
import pandas as pd
import ast
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader

from TrajectoryDataset import TrajectoryDataset
from models import TrajectoryModel
from GridGenerator import generate_spatial_grid
from logger import setup_logger, generate_ablation_name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_temporal_encoding', action='store_true')
    parser.add_argument('--use_time_embeddings', action='store_true')
    parser.add_argument('--use_spatial_fusion', action='store_true')
    parser.add_argument('--use_traj_traj_cl', action='store_true')
    parser.add_argument('--use_traj_node_cl', action='store_true')
    parser.add_argument('--use_node_node_cl', action='store_true')
    parser.add_argument('--contrastive_type', default='infonce')
    parser.add_argument('--epochs', type=int, default=30)
    return parser.parse_args()


def train(model, dataloader, graph_data, spatial_grid, optimizer, config, device, epoch=None, logger=None):
    model.train()
    total_losses = [0.0] * 4
    num_batches = len(dataloader)

    x, edge_index = graph_data
    x, edge_index = x.to(device), edge_index.to(device)
    spatial_grid = spatial_grid.to(device)

    pbar = tqdm(dataloader, desc=f"[Epoch {epoch}] Training", leave=True)
    for batch_idx, batch in enumerate(pbar):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        L_mtm, L_t2t, L_t2n, L_n2n = model(batch, spatial_grid)
        loss = config['λ1'] * L_mtm
        total_losses[0] += L_mtm.item()
        if config['use_traj_traj_cl']:
            loss += config['λ2'] * L_t2t
            total_losses[1] += L_t2t.item()
        if config['use_traj_node_cl']:
            loss += config['λ3'] * L_t2n
            total_losses[2] += L_t2n.item()
        if config['use_node_node_cl']:
            loss += config['λ4'] * L_n2n
            total_losses[3] += L_n2n.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "MTM": f"{L_mtm.item():.4f}",
            "T2T": f"{L_t2t.item():.4f}",
            "T2N": f"{L_t2n.item():.4f}",
            "N2N": f"{L_n2n.item():.4f}"
        })

    avg_losses = [l / num_batches for l in total_losses]
    logger.info(f"[Train Epoch {epoch}] MTM={avg_losses[0]:.4f}, T2T={avg_losses[1]:.4f}, T2N={avg_losses[2]:.4f}, N2N={avg_losses[3]:.4f}")
    return sum(avg_losses)


@torch.no_grad()
def validate(model, dataloader, graph_data, spatial_grid, config, device, epoch, logger=None):
    model.eval()
    total_losses = [0.0] * 4
    num_batches = len(dataloader)

    x, edge_index = graph_data
    x, edge_index = x.to(device), edge_index.to(device)
    spatial_grid = spatial_grid.to(device)

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        L_mtm, L_t2t, L_t2n, L_n2n = model(batch, spatial_grid)
        total_losses[0] += L_mtm.item()
        total_losses[1] += L_t2t.item()
        total_losses[2] += L_t2n.item()
        total_losses[3] += L_n2n.item()

    avg_losses = [l / num_batches for l in total_losses]
    logger.info(f"[Val Epoch {epoch}] MTM={avg_losses[0]:.4f}, T2T={avg_losses[1]:.4f}, T2N={avg_losses[2]:.4f}, N2N={avg_losses[3]:.4f}")
    return sum(avg_losses)

def load_trajectory_data_with_cache(data_path, include_files=None, cache_name="train_trajs.pt", use_one_file=False):
    cache_path = os.path.join(data_path, cache_name)

    if os.path.exists(cache_path):
        print(f"⚡ Loading cached trajectories from {cache_path}")
        return torch.load(cache_path)

    print("📄 Parsing CSVs (this may take time, but will be cached)...")

    all_files = sorted([
        os.path.join(data_path, f)
        for f in os.listdir(data_path)
        if f.endswith('.csv') and f != 'edge_features.csv'
    ])

    if include_files is not None:
        all_files = [os.path.join(data_path, f) for f in include_files]

    dfs = []
    for file_path in all_files:
        print(f"Loading file: {file_path}")
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

    # Save for fast reuse
    torch.save(trajectories, cache_path)
    print(f"✅ Saved parsed trajectories to {cache_path}")
    return trajectories

def load_edge_data(edge_features_path, edge_index_path):
    edge_features_df = pd.read_csv(edge_features_path)
    edge_index = np.load(edge_index_path)
    return edge_features_df, edge_index

def load_latest_checkpoint(model, optimizer, checkpoint_dir="checkpoints", device="cpu"):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if re.match(r"model_epoch_\d+\.pt", f)]

    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}. Starting from scratch.")
        return 1, model, optimizer

    checkpoint_files.sort(key=lambda f: int(re.search(r"model_epoch_(\d+)\.pt", f).group(1)))
    latest_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[-1])

    try:
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
        return start_epoch, model, optimizer

    except Exception as e:
        print(f"Error loading checkpoint {latest_checkpoint_path}: {e}")
        print("Starting training from scratch.")
        return 1, model, optimizer

# === Replace your __main__ block with this: ===
if __name__ == "__main__":
    args = parse_args()

    data_path = "datasets/didi_chengdu"
    edge_features_path = os.path.join(data_path, "edge_features.csv")
    edge_index_path = os.path.join(data_path, "line_graph_edge_idx.npy")
    dicts_pkl_path = os.path.join(data_path, "dicts.pkl")

    # === Load data ===
    edge_features_df, edge_index = load_edge_data(edge_features_path, edge_index_path)
    all_csvs = sorted([f for f in os.listdir(data_path) if f.endswith(".csv") and f != "edge_features.csv"])
    train_csvs, val_csvs = all_csvs[:-1], all_csvs[-1:]
    train_trajs = load_trajectory_data_with_cache(data_path, include_files=train_csvs, cache_name="train_trajs.pt")
    val_trajs = load_trajectory_data_with_cache(data_path, include_files=val_csvs, cache_name="val_trajs.pt")

    train_dataset = TrajectoryDataset(train_trajs, edge_features_df)
    val_dataset = TrajectoryDataset(val_trajs, edge_features_df)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=val_dataset.collate_fn)

    node_features = torch.tensor(edge_features_df[[ 'oneway', 'lanes', 'highway_id', 'length_id', 'bridge', 'tunnel', 'road_speed', 'traj_speed' ]].values, dtype=torch.float32)

    map_bbox = [30.730, 30.6554, 104.127, 104.0397]
    graph_pkl_path = os.path.join(data_path, "osm_graph", "ChengDu.pkl")
    spatial_grid_np = generate_spatial_grid(edge_features_path, dicts_pkl_path, graph_pkl_path, map_bbox=map_bbox, grid_size=(64, 64))
    spatial_grid_tensor = torch.tensor(spatial_grid_np, dtype=torch.float32).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        'λ1': 1.0, 'λ2': 5.0, 'λ3': 5.0, 'λ4': 5.0,
        'use_temporal_encoding': args.use_temporal_encoding,
        'use_time_embeddings': args.use_time_embeddings,
        'use_spatial_fusion': args.use_spatial_fusion,
        'use_traj_traj_cl': args.use_traj_traj_cl,
        'use_traj_node_cl': args.use_traj_node_cl,
        'use_node_node_cl': args.use_node_node_cl,
        'contrastive_type': args.contrastive_type,
    }

    ablation_name = generate_ablation_name(config)
    logger = setup_logger("logs", config)

    model = TrajectoryModel(config, graph_data).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    start_epoch, model, optimizer = load_latest_checkpoint(model, optimizer, device=device)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train(model, train_loader, graph_data, spatial_grid_tensor, optimizer, config, device, epoch, logger)
        val_loss = validate(model, val_loader, graph_data, spatial_grid_tensor, config, device, epoch, logger)

        print(f"[Epoch {epoch}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        save_path = os.path.join("checkpoints", f"model_epoch_{epoch}_{ablation_name}.pt")
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }, save_path)
