import os
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import faiss

from main import load_edge_data
from models import TrajectoryModel

# === Setup ===
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


def load_similarity_data(data_path, file_list, padding_id, max_len=64, num_queries=5000, detour_rate=0.1):
    dfs = []
    for fname in file_list:
        df = pd.read_csv(os.path.join(data_path, fname))
        df['path'] = df['path'].map(eval)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df['path_len'] = df['path'].map(len)
    df = df[(df['path_len'] >= 5) & (df['path_len'] <= max_len)]

    x_arr = []
    for _, row in df.iterrows():
        path = row['path']
        path = path[:max_len] + [padding_id] * max(0, max_len - len(path))
        x_arr.append(path)

    indices = np.random.permutation(len(df))[:num_queries]
    q_arr = []
    for idx in indices:
        path = df.iloc[idx]['path']
        detour_pos = np.random.choice(len(path), int(len(path) * detour_rate), replace=False)
        altered = [np.random.randint(padding_id) if i in detour_pos else e for i, e in enumerate(path)]
        altered = altered[:max_len] + [padding_id] * max(0, max_len - len(altered))
        q_arr.append(altered)

    return (
        torch.tensor(x_arr, dtype=torch.long).to(device),
        torch.tensor(q_arr, dtype=torch.long).to(device),
        indices
    )


def evaluation(model, data_path, file_list, padding_id, max_len=64):
    print("\n--- Trajectory Similarity Search ---")
    db_data, query_data, query_ids = load_similarity_data(data_path, file_list, padding_id, max_len)

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


# === Entry Point ===
if __name__ == "__main__":
    print("\n=== Similar Trajectory Retrieval ===")

    data_path = "datasets/didi_chengdu"
    edge_features_path = os.path.join(data_path, "edge_features.csv")
    edge_index_path = os.path.join(data_path, "line_graph_edge_idx.npy")

    edge_features_df, edge_index = load_edge_data(edge_features_path, edge_index_path)
    node_features = torch.tensor(edge_features_df[[
        'oneway', 'lanes', 'highway_id', 'length_id',
        'bridge', 'tunnel', 'road_speed', 'traj_speed'
    ]].values, dtype=torch.float32)

    graph_data = (node_features, torch.tensor(edge_index, dtype=torch.long))

    all_csvs = sorted([f for f in os.listdir(data_path) if f.endswith(".csv") and f != "edge_features.csv"])
    task_files = all_csvs[-1:]

    checkpoint_dir = "checkpoints"
    ablation_tag = ""  # Set tag like "trajnode", "spatialfusion", etc.
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
            model.eval()

            evaluation(model, data_path, task_files, padding_id=config['num_nodes'])

        except Exception as e:
            print(f"Error loading or evaluating {checkpoint_file}: {e}")
