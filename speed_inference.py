import os
import re
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from GridGenerator import generate_spatial_grid
from models import TrajectoryModel
from main import load_edge_data

# Reproducibility
import random
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Regressor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.fc(x).squeeze(1)

def parse_config_from_filename(filename):
    config = {
        'use_node_embedding': False,
        'use_temporal_encoding': False,
        'use_time_embeddings': False,
        'use_spatial_fusion': False,
        'use_traj_traj_cl': False,
        'use_traj_node_cl': False,
        'use_node_node_cl': False,
        'contrastive_type': 'infonce'
    }
    filename = filename.lower()
    if "node_embed" in filename: config['use_node_embedding'] = True
    if "temporal" in filename: config['use_temporal_encoding'] = True
    if "timeemb" in filename: config['use_time_embeddings'] = True
    if "spatialfusion" in filename: config['use_spatial_fusion'] = True
    if "trajtraj" in filename: config['use_traj_traj_cl'] = True
    if "trajnode" in filename: config['use_traj_node_cl'] = True
    if "nodenode" in filename: config['use_node_node_cl'] = True
    if "jsd" in filename: config['contrastive_type'] = "jsd"
    return config

def evaluation(model, feature_df, spatial_grid_tensor=None, fold=5):
    print("\n--- Speed Inference (Regression) ---")
    model.eval()
    with torch.no_grad():
        if model.config.get('use_spatial_fusion') and spatial_grid_tensor is not None:
            x, _, _ = model.encode_graph(spatial_grid_tensor.to(device))
        else:
            x, _, _ = model.encode_graph()
    x = x.cpu()

    y = torch.tensor(feature_df['road_speed'].values, dtype=torch.float32)
    node_ids = feature_df['road_id'].values
    x = x[node_ids]

    kf = KFold(n_splits=fold, shuffle=True, random_state=42)
    y_preds, y_trues = [], []

    for train_idx, val_idx in kf.split(x):
        x_train, y_train = x[train_idx], y[train_idx]
        x_val, y_val = x[val_idx], y[val_idx]

        regressor = Regressor(x.shape[1]).to(device)
        opt = torch.optim.Adam(regressor.parameters(),
                               lr=model.config.get("learning_rate", 1e-3),
                               weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min',
            factor=model.config.get("scheduler_factor", 0.8),
            patience=model.config.get("scheduler_patience", 10),
            threshold=1e-3, min_lr=1e-5
        )

        best_mse = float('inf')
        best_pred = None
        patience_counter = 0
        early_stop_patience = model.config.get("early_stop_patience", 200)

        for epoch in range(1, 10000):
            regressor.train()
            opt.zero_grad()
            pred = regressor(x_train.to(device))
            loss = nn.MSELoss()(pred, y_train.to(device))
            loss.backward()
            opt.step()

            regressor.eval()
            y_pred = regressor(x_val.to(device)).detach().cpu()
            mse = mean_squared_error(y_val, y_pred)
            scheduler.step(mse)

            if mse < best_mse:
                best_mse = mse
                best_pred = y_pred
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}. Best MSE: {best_mse:.4f}")
                break

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
    dicts_pkl_path = os.path.join(data_path, "dicts.pkl")
    graph_pkl_path = os.path.join(data_path, "osm_graph", "ChengDu.pkl")

    spatial_grid_np = generate_spatial_grid(
        edge_features_path,
        dicts_pkl_path,
        graph_pkl_path,
        map_bbox=[30.730, 30.6554, 104.127, 104.0397],
        grid_size=(64, 64)
    )
    spatial_grid_tensor = torch.tensor(spatial_grid_np, dtype=torch.float32).unsqueeze(0)

    checkpoint_dir = "checkpoints"
    checkpoint_files = [
        os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir)
        if f.endswith(".pt")
    ]
    checkpoint_files.sort(key=lambda f: int(re.search(r"model_epoch_(\d+)", f).group(1)))

    if not checkpoint_files:
        print("❌ No checkpoints found.")
        exit()

    for ckpt_path in checkpoint_files:
        try:
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
                'early_stop_patience': 200,
            }
            config = {**config_base, **parse_config_from_filename(ckpt_path)}

            model = TrajectoryModel(config, graph_data).to(device)
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            evaluation(model, edge_features_df, spatial_grid_tensor if config['use_spatial_fusion'] else None)

        except Exception as e:
            print(f"❌ Error with {ckpt_path}: {e}")
