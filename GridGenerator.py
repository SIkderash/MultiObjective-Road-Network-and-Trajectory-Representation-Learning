import numpy as np
import pandas as pd
import pickle
import networkx as nx

def generate_spatial_grid(feature_csv, dict_pkl_path, graph_pkl_path, map_bbox, grid_size=(64, 64)):
    """
    Converts road features into a 2D spatial grid [C, H, W].
    """
    features_df = pd.read_csv(feature_csv)
    with open(dict_pkl_path, 'rb') as f:
        idx2edge = pickle.load(f)['idx2edge']
    with open(graph_pkl_path, 'rb') as f:
        G = pickle.load(f)  # original OSMNX graph

    H, W = grid_size
    lat_max, lat_min, lon_max, lon_min = map_bbox
    delta_lat = (lat_max - lat_min) / H
    delta_lon = (lon_max - lon_min) / W

    feat_cols = ['oneway', 'lanes', 'highway_id', 'length_id', 'bridge', 'tunnel', 'road_speed', 'traj_speed']
    C = len(feat_cols)

    spatial_grid = np.zeros((C, H, W))
    count_grid = np.zeros((H, W))

    for _, row in features_df.iterrows():
        road_id = int(row['road_id'])
        if road_id not in idx2edge:
            continue
        # print(f"idx2edge[{road_id}] = {idx2edge[road_id]}")

        u, v, k = idx2edge[road_id]

        if u not in G.nodes or v not in G.nodes:
            continue

        lat1, lon1 = G.nodes[u]['y'], G.nodes[u]['x']
        lat2, lon2 = G.nodes[v]['y'], G.nodes[v]['x']
        lat = (lat1 + lat2) / 2
        lon = (lon1 + lon2) / 2

        i = int((lat - lat_min) / delta_lat)
        j = int((lon - lon_min) / delta_lon)

        if 0 <= i < H and 0 <= j < W:
            feat_vec = np.array([row[col] for col in feat_cols], dtype=np.float32)
            spatial_grid[:, i, j] += feat_vec
            count_grid[i, j] += 1

    for i in range(H):
        for j in range(W):
            if count_grid[i, j] > 0:
                spatial_grid[:, i, j] /= count_grid[i, j]

    return spatial_grid


def load_node_id_to_coord(dict_pkl_path, graph_pkl_path, map_bbox, grid_size=(64, 64)):

    with open(dict_pkl_path, 'rb') as f:
        idx2edge = pickle.load(f)['idx2edge']
    with open(graph_pkl_path, 'rb') as f:
        G = pickle.load(f)

    H, W = grid_size
    lat_max, lat_min, lon_max, lon_min = map_bbox
    delta_lat = (lat_max - lat_min) / H
    delta_lon = (lon_max - lon_min) / W

    node_id_to_coord = {}
    for road_id, (u, v, k) in idx2edge.items():
        if u not in G.nodes or v not in G.nodes:
            continue

        lat1, lon1 = G.nodes[u]['y'], G.nodes[u]['x']
        lat2, lon2 = G.nodes[v]['y'], G.nodes[v]['x']
        lat = (lat1 + lat2) / 2
        lon = (lon1 + lon2) / 2

        i = int((lat - lat_min) / delta_lat)
        j = int((lon - lon_min) / delta_lon)

        if 0 <= i < H and 0 <= j < W:
            node_id_to_coord[road_id] = (i, j)

    return node_id_to_coord

