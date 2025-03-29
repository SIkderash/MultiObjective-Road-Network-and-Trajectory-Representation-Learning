import torch
from torch.utils.data import Dataset
import random
from datetime import datetime

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

        # Fix: Ensure timestamps align with path (truncate if longer, pad if shorter)
        if len(timestamps) == len(path) + 1:
            timestamps = timestamps[:-1]  # drop last

        path = path[:self.max_len]
        timestamps = timestamps[:self.max_len]

        pad_len = self.max_len - len(path)
        path += [0] * pad_len

        if len(timestamps) == 0:
            timestamps = [0] * self.max_len
        else:
            timestamps += [timestamps[-1]] * pad_len

        hours, weekdays = [], []
        for ts in timestamps:
            dt = datetime.utcfromtimestamp(ts)
            hours.append(dt.hour)
            weekdays.append(dt.weekday())

        return {
            'road_nodes': torch.tensor(path, dtype=torch.long),
            'timestamps': torch.tensor(timestamps, dtype=torch.float),
            'hour': torch.tensor(hours, dtype=torch.long),
            'weekday': torch.tensor(weekdays, dtype=torch.long),
        }

    def collate_fn(self, batch):
        B = len(batch)
        L = self.max_len

        road_nodes = torch.stack([b['road_nodes'] for b in batch], dim=0)
        timestamps = torch.stack([b['timestamps'] for b in batch], dim=0)
        hours = torch.stack([b['hour'] for b in batch], dim=0)
        weekdays = torch.stack([b['weekday'] for b in batch], dim=0)

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
            'mtm_labels': mtm_labels,
            'timestamps': timestamps,
            'hour': hours,
            'weekday': weekdays
        }
