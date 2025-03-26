import os
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import faiss

from main import load_edge_data
from models import TrajectoryModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_similarity_data(data_path, file_list, padding_id, max_len=64, num_queries=5000, detour_rate=0.1):
    dfs = []
    for fname in file_list:
        df = pd.read_csv(os.path.join(data_path, fname))
        df['path'] = df['path'].map(eval)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df['path_len'] = df['path'].map(len)
    df = df[(df['path_len'] >= 5) & (df['path_len'] <= max_len)]

    # Database
    x_arr = []
    for _, row in df.iterrows():
        path = row['path']
        if len(path) > max_len:
            path = path[:max_len]
        else:
            path = path + [padding_id] * (max_len - len(path))
        x_arr.append(path)

    # Queries with detour
    detour = lambda rate=0.9: np.random.randint(padding_id) if np.random.rand() > rate else padding_id
    indices = np.random.permutation(len(df))[:num_queries]
    q_arr = []
    for idx in indices:
        path = df.iloc[idx]['path']
        detour_pos = np.random.choice(len(path), int(len(path) * detour_rate), replace=False)
        altered = [detour() if i in detour_pos else e for i, e in enumerate(path)]
        if len(altered) > max_len:
            altered = altered[:max_len]
        else:
            altered += [padding_id] * (max_len - len(altered))
        q_arr.append(altered)

    x_tensor = torch.tensor(x_arr, dtype=torch.long).to(device)
    q_tensor = torch.tensor(q_arr, dtype=torch.long).to(device)
    return x_tensor, q_tensor, indices


def evaluation(model, data_path, file_list, num_nodes, max_len=64):
    print("\n--- Trajectory Similarity Search ---")
    db_data, query_data, query_ids = load_similarity_data(
        data_path, file_list, padding_id=num_nodes, max_len=max_len
    )

    model.eval()
    with torch.no_grad():
        db_embed = model.encode_sequence(db_data).detach().cpu().numpy()
        query_embed = model.encode_sequence(query_data).detach().cpu().numpy()

    index = faiss.IndexFlatL2(db_embed.shape[1])
    index.add(db_embed)

    D, I = index.search(query_embed, k=10000)
    hits, ranks, no_hits = 0, 0, 0

    for i, row in enumerate(I):
        true_id = query_ids[i]
        if true_id in row:
            rank = np.where(row == true_id)[0][0]
            ranks += rank
            if rank < 10:
                hits += 1
        else:
            no_hits += 1

    num_queries = len(query_ids)
    print(f"Mean Rank: {ranks / num_queries:.2f}")
    print(f"Hit Rate@10: {hits / (num_queries - no_hits):.4f}")
    print(f"No-Hit Queries: {no_hits}/{num_queries}")


# === Main Entry ===
if __name__ == "__main__":
    print("\n=== Similar Trajectory Retrieval ===")
    data_path = "datasets/didi_chengdu"
    edge_features_path = os.path.join(data_path, "edge_features.csv")
    edge_index_path = os.path.join(data_path, "line_graph_edge_idx.npy")

    edge_features_df, edge_index = load_edge_data(edge_features_path, edge_index_path)
    node_features = torch.tensor(edge_features_df[[
        'oneway', 'lanes', 'highway_id', 'length_id',
        'bridge', 'tunnel', 'road_speed', 'traj_speed'
    ]].values, dtype=torch.float32).to(device)

    graph_data = (node_features, torch.tensor(edge_index, dtype=torch.long).to(device))
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

    model = TrajectoryModel(config, graph_data).to(device)

    checkpoint_dir="checkpoints"
    checkpoint_files = []
    for filename in os.listdir(checkpoint_dir):
        if re.match(r"model_epoch_\d+\.pt", filename):
            checkpoint_files.append(os.path.join(checkpoint_dir, filename))

    if not checkpoint_files:
        print(f"No model_epoch_*.pt files found in {checkpoint_dir}")
        exit()

    checkpoint_files.sort(key=lambda f: int(re.search(r"model_epoch_(\d+)\.pt", f).group(1))) # Sort by epoch number

    for checkpoint_file in checkpoint_files:
        try:
            print(f"Loading checkpoint: {checkpoint_file}")
            model = TrajectoryModel(config, graph_data)
            checkpoint = torch.load(checkpoint_file)

            state_dict = checkpoint['model_state_dict']
            new_state_dict = {}
            for key, val in state_dict.items():
                if key.endswith('gat1.lin.weight'):
                    new_state_dict['gat.gat1.lin_src.weight'] = val
                    new_state_dict['gat.gat1.lin_dst.weight'] = val
                elif key.endswith('gat2.lin.weight'):
                    new_state_dict['gat.gat2.lin_src.weight'] = val
                    new_state_dict['gat.gat2.lin_dst.weight'] = val
                else:
                    new_state_dict[key] = val

            model.load_state_dict(new_state_dict)
            model = model.eval()
            all_csvs = sorted([f for f in os.listdir(data_path) if f.endswith(".csv") and f != "edge_features.csv"])
            task_files = all_csvs[-1:]  # last file for eval
            evaluation(model, data_path, task_files, config['num_nodes'], max_len=64)
        except Exception as e:
            print(f"Error loading or evaluating {checkpoint_file}: {e}")
