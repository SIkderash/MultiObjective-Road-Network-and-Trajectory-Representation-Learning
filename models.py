import math
import random
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# ==== GAT Module ====
class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=0.2)
        self.gat2 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=0.2)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return x  # [num_nodes, embed_dim]


# ==== CNN Module ====
class CNNModel(nn.Module):
    def __init__(self, in_channels=8, out_channels=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)  # [B, C_out, H, W]
    
class TemporalEncoding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.omega = nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, embed_dim)).float(), requires_grad=True
        )
        self.bias = nn.Parameter(torch.zeros(embed_dim), requires_grad=True)
        self.div_term = math.sqrt(1.0 / embed_dim)

    def forward(self, timestamps):  # [B, L]
        time_encode = timestamps.unsqueeze(-1) * self.omega + self.bias
        return self.div_term * torch.cos(time_encode)  # [B, L, D]



# ==== Spatial Transformer ====
class SpatialTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer(x)  # [B, S, D]

class TrajectoryTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, query, key, value):
        return self.transformer(tgt=query, memory=key)  # [B, L, D]


# ==== MTM Head ====
class MTMHead(nn.Module):
    def __init__(self, embed_dim, num_nodes):
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_nodes)
        self.dropout = nn.Dropout(0.1)
        self.loss_func = nn.CrossEntropyLoss()
        self.num_nodes = num_nodes

    def forward(self, x, **kwargs):
        targets = kwargs['origin_nodes'].reshape(-1)
        logits = self.linear(self.dropout(x)).reshape(-1, self.num_nodes)
        return self.loss_func(logits, targets)


# ==== MTD Head ====
class MTDHead(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(0.1)
        self.loss_func = nn.MSELoss()

    def forward(self, x, **kwargs):
        targets = kwargs['origin_deltas'].reshape(-1)
        preds = self.linear(self.dropout(x)).squeeze(-1)
        return self.loss_func(preds, targets)


# ==== NSP Head ====
class NSPHead(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 2)
        self.dropout = nn.Dropout(0.1)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, **kwargs):
        targets = kwargs['labels']
        logits = self.linear(self.dropout(x))  # [B, 2]
        return self.loss_func(logits, targets)
    

class MaskedHour(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 24)
        self.dropout = nn.Dropout(0.1)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, **kwargs):
        labels = kwargs['origin_hour'].reshape(-1)
        logits = self.linear(self.dropout(x)).reshape(-1, 24)
        return self.loss_func(logits, labels)


class MaskedWeekday(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 7)
        self.dropout = nn.Dropout(0.1)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, **kwargs):
        labels = kwargs['origin_weekday'].reshape(-1)
        logits = self.linear(self.dropout(x)).reshape(-1, 7)
        return self.loss_func(logits, labels)
    

class ContrastiveHead(nn.Module):
    def __init__(self, embed_dim, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, embeds):
        """
        embeds: [2B, D] where first B are anchors, second B are positives
        embeds: [2B, D] where first B are anchors, second B are positives
        """
        z = F.normalize(self.projection(embeds), dim=1)  # [2B, D]
        B = z.size(0) // 2

        # Similarity matrix (excluding self)
        sim = torch.matmul(z, z.T) / self.temperature  # [2B, 2B]
        mask = torch.eye(2 * B, device=embeds.device, dtype=torch.bool)
        sim.masked_fill_(mask, -1e9)

        # Construct targets
        targets = torch.arange(B, 2 * B, device=embeds.device)
        targets = torch.cat([targets, torch.arange(0, B, device=embeds.device)], dim=0)
        B = z.size(0) // 2

        # Similarity matrix (excluding self)
        sim = torch.matmul(z, z.T) / self.temperature  # [2B, 2B]
        mask = torch.eye(2 * B, device=embeds.device, dtype=torch.bool)
        sim.masked_fill_(mask, -1e9)

        # Construct targets
        targets = torch.arange(B, 2 * B, device=embeds.device)
        targets = torch.cat([targets, torch.arange(0, B, device=embeds.device)], dim=0)

        return self.loss_func(sim, targets)






# # ==== Full Model ====
# class TrajectoryModel(nn.Module):
#     def __init__(self, config, graph_data):
#         super().__init__()
#         self.gat = GATModel(config['node_feat_dim'], config['gat_hidden'], config['embed_dim'])
#         self.cnn = CNNModel(config['spatial_in_channels'], config['embed_dim'])
#         # self.temporal_encoding = TemporalEncoding(embed_dim=config['embed_dim'])

#         self.spatial_transformer = SpatialTransformer(config['embed_dim'], config['num_heads'], config['spatial_layers'])
#         self.traj_transformer = TrajectoryTransformer(config['embed_dim'], config['num_heads'], config['traj_layers'])

#         # Learnable mask token
#         self.mask_token = nn.Parameter(torch.zeros(1, config['embed_dim']))

#         # Heads
#         self.mtm = MTMHead(config['embed_dim'], config['num_nodes'])
#         self.mtd = MTDHead(config['embed_dim'])
#         self.hour_head = MaskedHour(config['embed_dim'])
#         self.weekday_head = MaskedWeekday(config['embed_dim'])

#         # Save graph data for reuse
#         self._x, self._edge_index = graph_data

#     def encode_graph(self):
#         self.eval()
#         with torch.no_grad():
#             device = next(self.parameters()).device
#             x = self._x.to(device)
#             edge_index = self._edge_index.to(device)
#             node_embeddings = self.gat(x, edge_index)
#             return node_embeddings

#     def forward(self, batch, spatial_grid):
#         device = next(self.parameters()).device
#         B, L = batch['road_nodes'].shape

#         # === GAT ===
#         x = self._x.to(device)
#         edge_index = self._edge_index.to(device)
#         node_embeddings = self.gat(x, edge_index)  # [N, D]

#         node_embeddings = torch.cat([node_embeddings, self.mask_token.to(device)], dim=0)
#         road_embeddings = node_embeddings[batch['road_nodes']]  # [B, L, D]

#         # === Spatial Transformer ===
#         spatial_features = self.cnn(spatial_grid.to(device))  # [1, D, H, W]
#         spatial_tokens = spatial_features.flatten(2).transpose(1, 2)  # [1, S, D]
#         spatial_repr = self.spatial_transformer(spatial_tokens).expand(B, -1, -1)  # [B, S, D]

#         # === Trajectory Transformer ===
#         fused_repr = self.traj_transformer(road_embeddings, spatial_repr, spatial_repr)  # [B, L, D]

#         # === Heads ===
#         mtm_x = fused_repr[batch['mtm_mask']]
#         mtd_x = fused_repr[batch['mtd_mask']]
#         hour_x = fused_repr[batch['mtm_mask']]
#         weekday_x = fused_repr[batch['mtm_mask']]

#         # Losses
#         L_mtm = self.mtm(mtm_x, origin_nodes=batch['mtm_labels'])
#         L_mtd = self.mtd(mtd_x, origin_deltas=batch['mtd_labels'])
#         L_hour = self.hour_head(hour_x, origin_hour=batch['hour_labels'])
#         L_weekday = self.weekday_head(weekday_x, origin_weekday=batch['weekday_labels'])

#         return L_mtm, L_mtd, L_hour, L_weekday


#         # === Add Temporal Encoding ===
#         # timestamps = batch['timestamps'][:, :-1].float()  # shape [B, L]
#         # time_repr = self.temporal_encoding(timestamps)    # shape [B, L, D]
#         # road_embeddings += time_repr  # Add temporal context




class TrajectoryModel(nn.Module):
    def __init__(self, config, graph_data):
        super().__init__()
        self.gat = GATModel(config['node_feat_dim'], config['gat_hidden'], config['embed_dim'])
        self.cnn = CNNModel(config['spatial_in_channels'], config['embed_dim'])
        self.spatial_transformer = SpatialTransformer(config['embed_dim'], config['num_heads'], config['spatial_layers'])
        self.traj_transformer = TrajectoryTransformer(config['embed_dim'], config['num_heads'], config['traj_layers'])

        self.mask_token = nn.Parameter(torch.zeros(1, config['embed_dim']))
        self.mtm = MTMHead(config['embed_dim'], config['num_nodes'])
        self.contrastive_head = ContrastiveHead(config['embed_dim'])

        self._x, self._edge_index = graph_data

    def forward(self, batch, spatial_grid):
        device = next(self.parameters()).device
        B, L = batch['road_nodes'].shape

        x = self._x.to(device)
        edge_index = self._edge_index.to(device)
        node_embeddings = self.gat(x, edge_index)
        node_embeddings = torch.cat([node_embeddings, self.mask_token.to(device)], dim=0)

        # === View 1 ===
        # === View 1 ===
        road_embeddings = node_embeddings[batch['road_nodes']]
        spatial_features = self.cnn(spatial_grid.to(device))
        spatial_tokens = spatial_features.flatten(2).transpose(1, 2)
        spatial_repr = self.spatial_transformer(spatial_tokens).expand(B, -1, -1)
        fused_repr = self.traj_transformer(road_embeddings, spatial_repr, spatial_repr)

        mtm_x = fused_repr[batch['mtm_mask']]
        L_mtm = self.mtm(mtm_x, origin_nodes=batch['mtm_labels'])

        # === View 2 (Augmented for contrastive) ===
        road_nodes_pos = batch['road_nodes'].clone()
        mtm_mask_pos = torch.zeros_like(road_nodes_pos, dtype=torch.bool)
        for i in range(B):
            num_mask = max(1, int(0.15 * L))
            mask_indices = random.sample(range(L), num_mask)
            mtm_mask_pos[i, mask_indices] = 1
        road_nodes_pos[mtm_mask_pos] = self._x.shape[0]  # [MASK] token index

        road_embed_pos = node_embeddings[road_nodes_pos]
        fused_pos = self.traj_transformer(road_embed_pos, spatial_repr, spatial_repr)

        traj_embed = fused_repr.mean(dim=1)        # [B, D]
        traj_embed_pos = fused_pos.mean(dim=1)     # [B, D]
        concat_embed = torch.cat([traj_embed, traj_embed_pos], dim=0)  # [2B, D]
        L_contrastive = self.contrastive_head(concat_embed)

        return L_mtm, L_contrastive

