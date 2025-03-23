# import torch
# from torch.utils.data import Dataset
# import random

# class TrajectoryDataset(Dataset):
#     def __init__(self, trajectories, edge_features_df, max_len=64):
#         self.trajectories = trajectories
#         self.edge_features_df = edge_features_df
#         self.max_len = max_len
#         self.num_nodes = edge_features_df['road_id'].nunique()  # Used to compute MASK_TOKEN_ID

#     def __len__(self):
#         return len(self.trajectories)

#     def __getitem__(self, idx):
#         traj = self.trajectories[idx]
#         path = traj['path']
#         timestamps = traj['timestamps']
#         pass_time = traj['pass_time']

#         # Truncate or pad trajectory
#         if len(path) > self.max_len:
#             path = path[:self.max_len]
#             timestamps = timestamps[:self.max_len + 1]
#             pass_time = pass_time[:self.max_len]
#         else:
#             pad_len = self.max_len - len(path)
#             path += [0] * pad_len
#             pass_time += [0] * pad_len
#             timestamps += [timestamps[-1]] * pad_len

#         return {
#             'road_nodes': torch.tensor(path, dtype=torch.long),
#             'timestamps': torch.tensor(timestamps, dtype=torch.float),
#             'pass_time': torch.tensor(pass_time, dtype=torch.float)
#         }

#     def collate_fn(self, batch):
#         """
#         Returns a batch dictionary for training with MTM, MTD, and temporal classification (hour/weekday).
#         Shapes:
#             - road_nodes: [B, L]
#             - timestamps: [B, L]
#             - mtm_mask: [B, L] (bool)
#             - mtm_labels: [num_masked]
#             - mtd_labels: [num_masked]
#             - hour_labels: [num_masked]
#             - weekday_labels: [num_masked]
#         """
#         B = len(batch)
#         L = self.max_len

#         road_nodes = torch.stack([b['road_nodes'] for b in batch], dim=0)     # [B, L]
#         timestamps = torch.stack([b['timestamps'] for b in batch], dim=0)     # [B, L+1]
#         pass_time = torch.stack([b['pass_time'] for b in batch], dim=0)       # [B, L]

#         # === MTM: Mask random positions ===
#         mtm_mask = torch.zeros_like(road_nodes, dtype=torch.bool)
#         for i in range(B):
#             num_mask = max(1, int(0.15 * L))
#             mask_indices = random.sample(range(L), num_mask)
#             mtm_mask[i, mask_indices] = 1

#         mtm_labels = road_nodes.masked_select(mtm_mask).clone()  # [num_masked]

#         # Replace masked input with special [MASK] token
#         MASK_TOKEN_ID = self.num_nodes  # Usually last index in embedding
#         road_nodes_masked = road_nodes.clone()
#         road_nodes_masked[mtm_mask] = MASK_TOKEN_ID

#         # === MTD: Time Delta Prediction ===
#         deltas = (timestamps[:, 1:] - timestamps[:, :-1]) / 100.0  # [B, L]
#         mtd_labels = deltas.masked_select(mtm_mask)               # [num_masked]

#         # === Temporal Classification (Hour & Weekday) ===
#         ts_main = timestamps[:, :-1]  # [B, L]
#         hour_labels = ((ts_main % (24 * 60 * 60)) // 3600).long()       # [B, L]
#         weekday_labels = ((ts_main // (24 * 60 * 60)) % 7).long()       # [B, L]

#         hour_labels = hour_labels.masked_select(mtm_mask)
#         weekday_labels = weekday_labels.masked_select(mtm_mask)

#         # === Validations (assertions) ===
#         assert hour_labels.numel() > 0, "No masked hour labels!"
#         assert weekday_labels.numel() > 0, "No masked weekday labels!"
#         assert hour_labels.max() < 24 and hour_labels.min() >= 0, f"Hour labels out of range: {hour_labels.tolist()}"
#         assert weekday_labels.max() < 7 and weekday_labels.min() >= 0, f"Weekday labels out of range: {weekday_labels.tolist()}"

#         return {
#             'road_nodes': road_nodes_masked,
#             'timestamps': ts_main,        # still useful if you re-enable temporal encoding later
#             'mtm_mask': mtm_mask,
#             'mtm_labels': mtm_labels,
#             'mtd_labels': mtd_labels,
#             'mtd_mask': mtm_mask,         # same mask as MTM
#             'hour_labels': hour_labels,
#             'weekday_labels': weekday_labels
#         }



import torch
from torch.utils.data import Dataset
import random

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, edge_features_df, max_len=64):
        self.trajectories = trajectories
        self.edge_features_df = edge_features_df
        self.max_len = max_len
        self.num_nodes = edge_features_df['road_id'].nunique()

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        path = traj['path']
        timestamps = traj['timestamps']

        if len(path) > self.max_len:
            path = path[:self.max_len]
            timestamps = timestamps[:self.max_len + 1]
        else:
            pad_len = self.max_len - len(path)
            path += [0] * pad_len
            timestamps += [timestamps[-1]] * pad_len

        return {
            'road_nodes': torch.tensor(path, dtype=torch.long),
            'timestamps': torch.tensor(timestamps, dtype=torch.float),
        }

    def collate_fn(self, batch):
        B = len(batch)
        L = self.max_len

        road_nodes = torch.stack([b['road_nodes'] for b in batch], dim=0)
        mtm_mask = torch.zeros_like(road_nodes, dtype=torch.bool)

        for i in range(B):
            num_mask = max(1, int(0.15 * L))
            mask_indices = random.sample(range(L), num_mask)
            mtm_mask[i, mask_indices] = 1

        mtm_labels = road_nodes.masked_select(mtm_mask).clone()

        MASK_TOKEN_ID = self.num_nodes
        road_nodes_masked = road_nodes.clone()
        road_nodes_masked[mtm_mask] = MASK_TOKEN_ID

        return {
            'road_nodes': road_nodes_masked,
            'mtm_mask': mtm_mask,
            'mtm_labels': mtm_labels
        }


