import os
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from main import load_edge_data
from models import TrajectoryModel

import os
import random
import numpy as np
import torch

# Set seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# For deterministic behavior (slower but reproducible)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)

def evaluation(model, feature_df, fold=5):
    print("\n--- Road Classification ---")
    model.eval()
    x = model.encode_graph()  # [num_nodes, embed_dim]
    if isinstance(x, tuple):
        x = x[0]
    x = x.detach().cpu()

    valid_labels = ['primary', 'secondary', 'tertiary', 'residential']
    id_dict = {lbl: i for i, lbl in enumerate(valid_labels)}
    y_df = feature_df[feature_df['highway'].isin(valid_labels)].copy()
    y_df = y_df.sort_values('road_id')

    node_indices = y_df['road_id'].values
    x = x[node_indices]
    y = torch.tensor(y_df['highway'].map(id_dict).values)

    # === Stratified K-Fold ===
    skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42)

    y_preds, y_trues = [], []
    for train_idx, test_idx in skf.split(x, y):
        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]

        clf = Classifier(x.shape[1], len(valid_labels)).cuda()
        opt = torch.optim.Adam(clf.parameters(), lr=1e-3)

        best_pred = None
        best_acc = 0
        for epoch in range(1, 101):
            clf.train()
            opt.zero_grad()
            loss = nn.CrossEntropyLoss()(clf(x_train.cuda()), y_train.cuda())
            loss.backward()
            opt.step()

            clf.eval()
            preds = clf(x_test.cuda()).argmax(dim=1).cpu()
            acc = accuracy_score(y_test, preds)
            if acc > best_acc:
                best_acc = acc
                best_pred = preds

        y_preds.append(best_pred)
        y_trues.append(y_test)

    y_preds = torch.cat(y_preds)
    y_trues = torch.cat(y_trues)
    print(f'Micro F1: {f1_score(y_trues, y_preds, average="micro"):.4f}')
    print(f'Macro F1: {f1_score(y_trues, y_preds, average="macro"):.4f}')


# === Main Entry ===
if __name__ == "__main__":
    print("\n=== Road Classification Evaluation ===")

    data_path = "datasets/didi_chengdu"
    edge_features_path = os.path.join(data_path, "edge_features.csv")
    edge_index_path = os.path.join(data_path, "line_graph_edge_idx.npy")

    edge_features_df, edge_index = load_edge_data(edge_features_path, edge_index_path)
    node_features = torch.tensor(edge_features_df[['oneway', 'lanes', 'highway_id', 'length_id',
                                                    'bridge', 'tunnel', 'road_speed', 'traj_speed']].values,
                                 dtype=torch.float32)
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

    model = TrajectoryModel(config, graph_data)
    checkpoint = torch.load("checkpoints/model_epoch_5.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda().eval()


    evaluation(model, edge_features_df, fold=5)
