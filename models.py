# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import math
import numpy as np

# === Positional Encoding ===
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# === Temporal Encoding ===
class TemporalEncoding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.omega = nn.Parameter(torch.from_numpy(1 / 10 ** np.linspace(0, 9, embed_dim)).float(), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(embed_dim), requires_grad=True)
        self.div_term = math.sqrt(1.0 / embed_dim)

    def forward(self, timestamps):
        time_encode = timestamps.unsqueeze(-1) * self.omega + self.bias
        return self.div_term * torch.cos(time_encode)


# === GAT ===
class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=0.2)
        self.gat2 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=0.2)

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        return self.gat2(x, edge_index)


# === CNN ===
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
        return self.encoder(x)


# === Transformer ===
class SpatialTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)

    def forward(self, x):
        return self.encoder(x)

def contrastive_loss(z1, z2, loss_type="infonce", temperature=0.1):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    if loss_type == "infonce":
        z = torch.cat([z1, z2], dim=0)
        sim = torch.matmul(z, z.T) / temperature
        labels = torch.arange(z1.size(0), device=z.device)
        labels = torch.cat([labels + z1.size(0), labels], dim=0)
        mask = torch.eye(2 * z1.size(0), device=z.device).bool()
        sim.masked_fill_(mask, -1e9)
        return F.cross_entropy(sim, labels)

    elif loss_type == "jsd":
        def D(p, q):
            sim = torch.mm(p, q.T)
            E_pos = math.log(2.) - F.softplus(-sim)
            E_neg = F.softplus(-sim) + sim - math.log(2.)
            return E_pos.mean(), E_neg.mean()
        E_pos, E_neg = D(z1, z2)
        return E_neg - E_pos

    else:
        raise ValueError("Unsupported contrastive loss type")
    
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, query, key, value, spatial_mask=None):
        # Make sure mask shape is [B, key_len]
        key_padding_mask = None
        if spatial_mask is not None:
            key_padding_mask = ~spatial_mask.bool()  # invert: 1 = attend, 0 = pad
            # make sure it's bool and shape [B, key_len]
            assert key_padding_mask.shape[1] == key.shape[1], \
                f"Mask shape {key_padding_mask.shape} doesn't match key shape {key.shape}"

        output, _ = self.attn(query, key, value, key_padding_mask=key_padding_mask)
        return output


# === MTM Head ===
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

# === Full Model ===
class TrajectoryModel(nn.Module):
    def __init__(self, config, graph_data, node_id_to_coord=None):
        super().__init__()
        self.config = config
        if config.get("use_node_embedding", False):
            self.use_node_embedding = True
            self.node_embedding = nn.Embedding(config['num_nodes'], config['embed_dim'])
            nn.init.xavier_uniform_(self.node_embedding.weight)
        else:
            self.use_node_embedding = False
            self._x = graph_data[0]

        self._edge_index = graph_data[1]
        self.node_id_to_coord = node_id_to_coord
        in_channels = config['embed_dim'] if config['use_node_embedding'] else config['node_feat_dim']
        self.gat = GATModel(in_channels, config['gat_hidden'], config['embed_dim'])
        self.cnn = CNNModel(config['spatial_in_channels'], config['embed_dim'])
        self.spatial_transformer = SpatialTransformer(config['embed_dim'], config['num_heads'], config['spatial_layers'])

        self.temporal_encoding = TemporalEncoding(config['embed_dim']) if config['use_temporal_encoding'] else None
        self.hour_embed = nn.Embedding(24, config['embed_dim']) if config['use_time_embeddings'] else None
        self.weekday_embed = nn.Embedding(7, config['embed_dim']) if config['use_time_embeddings'] else None

        self.pos_encoder = PositionalEncoding(config['embed_dim'])
        self.cross_attn = CrossAttention(config['embed_dim'], config['num_heads'])

        self.mask_token = nn.Embedding(1, config['embed_dim'])
        nn.init.trunc_normal_(self.mask_token.weight, std=0.02)

        self.mtm = MTMHead(config['embed_dim'], config['num_nodes'])

        self.projector = nn.Sequential(
            nn.Linear(config['embed_dim'], config['embed_dim']),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config['embed_dim'], config['embed_dim'])
        )

        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(config['embed_dim'])
        self.final_norm = nn.LayerNorm(config['embed_dim'])

        transformer_layer = nn.TransformerEncoderLayer(config['embed_dim'], config['num_heads'], batch_first=True)
        self.traj_encoder = nn.TransformerEncoder(transformer_layer, num_layers=config['traj_layers'])

    
    def encode_graph(self, spatial_grid=None):
        device = next(self.parameters()).device

        if self.use_node_embedding:
            x = self.node_embedding.weight
        else:
            x = self._x.to(device)

        edge_index = self._edge_index.to(device)
        node_embed = self.gat(x, edge_index)

        spatial_repr = None
        spatial_shape = None

        if self.config['use_spatial_fusion'] and spatial_grid is not None:
            spatial_features = self.cnn(spatial_grid)  # [B, D, H, W]
            spatial_shape = spatial_features.shape[2:]  # (H, W)
            spatial_tokens = spatial_features.flatten(2).transpose(1, 2)  # [B, HW, D]
            spatial_repr = self.spatial_transformer(spatial_tokens)

            context = spatial_repr.mean(dim=(0, 1))  # Global mean pooling
            context = self.norm(self.dropout(context)).unsqueeze(0)  # [1, D]
            node_embed = node_embed + context  # Residual fusion

        return node_embed, spatial_repr, spatial_shape



    def encode_sequence(self, node_sequences, timestamps=None, hour=None, weekday=None, 
                        spatial_repr=None, spatial_shape=None):
        device = next(self.parameters()).device
        node_embed = self.encode_graph(None)[0]  # Don't recompute, we already have it
        node_embed = torch.cat([node_embed, self.mask_token.weight], dim=0)

        traj_embed = node_embed[node_sequences.to(device)]

        if self.config['use_temporal_encoding'] and timestamps is not None:
            traj_embed += self.temporal_encoding(timestamps.to(device))
        if self.config['use_time_embeddings'] and hour is not None and weekday is not None:
            traj_embed += self.hour_embed(hour.to(device))
            traj_embed += self.weekday_embed(weekday.to(device))

        traj_embed = self.pos_encoder(traj_embed)
        traj_embed = self.dropout(traj_embed)

        if self.config['use_spatial_fusion'] and spatial_repr is not None and self.node_id_to_coord is not None:
            B, HW, D = spatial_repr.shape
            H, W = spatial_shape 
            orig_H, orig_W = 64, 64  # Used in node_id_to_coord
            scale_y, scale_x = H / orig_H, W / orig_W

            mask = torch.zeros((B, HW), device=device)
            for i in range(node_sequences.size(0)):
                for nid in node_sequences[i]:
                    nid = int(nid.item())
                    if nid in self.node_id_to_coord:
                        y, x = self.node_id_to_coord[nid]
                        y = int(y * scale_y)
                        x = int(x * scale_x)
                        if 0 <= y < H and 0 <= x < W:
                            idx = y * W + x
                            mask[i, idx] = 1.0

            mask = mask.bool()

            traj_embed = traj_embed + self.cross_attn(
                traj_embed,
                spatial_repr,
                spatial_repr,
                spatial_mask=mask
            )
            traj_embed = self.norm(self.dropout(traj_embed))

        encoded = self.final_norm(self.traj_encoder(traj_embed))

        return encoded


    def node_sequence_contrastive_loss(self, traj_embed, road_nodes_tensor, node_embeddings):
        B = traj_embed.size(0)
        node_repr = node_embeddings[road_nodes_tensor].mean(dim=1)
        traj_embed = F.normalize(traj_embed, dim=-1)
        node_repr = F.normalize(self.projector(node_repr), dim=-1)
        sim = torch.matmul(traj_embed, node_repr.T)
        labels = torch.arange(B, device=traj_embed.device)
        return F.cross_entropy(sim, labels)

    def node_node_contrastive_loss(self, node_embed):
        node_embed_2 = node_embed[torch.randperm(node_embed.size(0))]
        return contrastive_loss(node_embed, node_embed_2, loss_type=self.config['contrastive_type'])

    def forward(self, batch, spatial_grid=None):
        device = next(self.parameters()).device
        B, L = batch['road_nodes'].shape

        node_embeddings, spatial_repr, spatial_shape = self.encode_graph(spatial_grid)
        node_embeddings = torch.cat([node_embeddings, self.mask_token.weight.to(device)], dim=0)

        traj_repr = self.encode_sequence(
            node_sequences=batch['road_nodes'],
            timestamps=batch.get('timestamps'),
            hour=batch.get('hour'),
            weekday=batch.get('weekday'),
            spatial_repr=spatial_repr,
            spatial_shape=spatial_shape
        )

        if batch.get('attention_mask') is not None:
            pooled = (traj_repr * batch['attention_mask'].unsqueeze(-1)).sum(dim=1) / \
                    batch['attention_mask'].sum(dim=1, keepdim=True).clamp(min=1e-5)
        else:
            pooled = traj_repr.mean(dim=1)
    
        mtm_x = traj_repr[batch['mtm_mask']]
        L_mtm = self.mtm(mtm_x, origin_nodes=batch['mtm_labels'])

        # Contrastive losses
        L_cl_traj = torch.tensor(0.0, device=device)
        L_cl_node = torch.tensor(0.0, device=device)
        L_cl_node_node = torch.tensor(0.0, device=device)

        if self.config['use_traj_traj_cl']:
            traj_embed = self.projector(pooled)
            road_nodes_aug = batch['road_nodes'].clone()
            for i in range(B):
                num_mask = max(1, int(0.15 * L))
                mask_indices = torch.randperm(L)[:num_mask]
                road_nodes_aug[i, mask_indices] = self._x.shape[0]

            aug_repr = self.encode_sequence(
                node_sequences=road_nodes_aug,
                timestamps=batch.get('timestamps'),
                hour=batch.get('hour'),
                weekday=batch.get('weekday'),
                spatial_repr=spatial_repr,
                spatial_shape=spatial_shape
            )

            if batch.get('attention_mask') is not None:
                aug_pooled = (aug_repr * batch['attention_mask'].unsqueeze(-1)).sum(dim=1) / \
                            batch['attention_mask'].sum(dim=1, keepdim=True).clamp(min=1e-5)
            else:
                aug_pooled = aug_repr.mean(dim=1)

            traj_embed_aug = self.projector(aug_pooled)
            L_cl_traj = contrastive_loss(traj_embed, traj_embed_aug, loss_type=self.config['contrastive_type'])

        if self.config['use_traj_node_cl']:
            L_cl_node = self.node_sequence_contrastive_loss(traj_embed, batch['road_nodes'], node_embeddings)

        if self.config['use_node_node_cl']:
            L_cl_node_node = self.node_node_contrastive_loss(node_embeddings[:-1])

        return L_mtm, L_cl_traj, L_cl_node, L_cl_node_node