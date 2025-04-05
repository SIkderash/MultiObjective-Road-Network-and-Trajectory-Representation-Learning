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


# === Cross Attention ===
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, query, key, value):
        out, _ = self.attn(query, key, value)
        return out


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


# === Contrastive Loss ===
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


# === Full Model ===
class TrajectoryModel(nn.Module):
    def __init__(self, config, graph_data):
        super().__init__()
        self.config = config
        self._x, self._edge_index = graph_data

        self.gat = GATModel(config['node_feat_dim'], config['gat_hidden'], config['embed_dim'])
        self.cnn = CNNModel(config['spatial_in_channels'], config['embed_dim'])
        self.spatial_transformer = SpatialTransformer(config['embed_dim'], config['num_heads'], config['spatial_layers'])

        self.temporal_encoding = TemporalEncoding(config['embed_dim']) if config['use_temporal_encoding'] else None
        self.hour_embed = nn.Embedding(24, config['embed_dim']) if config['use_time_embeddings'] else None
        self.weekday_embed = nn.Embedding(7, config['embed_dim']) if config['use_time_embeddings'] else None

        self.pos_encoder = PositionalEncoding(config['embed_dim'])

        transformer_layer = nn.TransformerEncoderLayer(config['embed_dim'], config['num_heads'], batch_first=True)
        self.traj_encoder = nn.TransformerEncoder(transformer_layer, num_layers=config['traj_layers'])

        self.cross_attn = CrossAttention(config['embed_dim'], config['num_heads'])
        self.mask_token = nn.Parameter(torch.zeros(1, config['embed_dim']))

        self.projector = nn.Sequential(
            nn.Linear(config['embed_dim'], config['embed_dim']),
            nn.ReLU(),
            nn.Linear(config['embed_dim'], config['embed_dim'])
        )

        self.mtm = MTMHead(config['embed_dim'], config['num_nodes'])

    def encode_graph(self, spatial_grid=None):
        x = self._x.to(next(self.parameters()).device)
        edge_index = self._edge_index.to(next(self.parameters()).device)
        node_embed = self.gat(x, edge_index)

        if self.config['use_spatial_fusion'] and spatial_grid is not None:
            spatial_features = self.cnn(spatial_grid)
            spatial_tokens = spatial_features.flatten(2).transpose(1, 2)
            spatial_repr = self.spatial_transformer(spatial_tokens)
            context = spatial_repr.mean(dim=1)
            node_embed = node_embed + context

        return node_embed
    
    def encode_sequence(self, road_nodes_tensor, timestamps=None, hour=None, weekday=None, spatial_grid=None):
        device = next(self.parameters()).device
        node_embed = self.encode_graph(spatial_grid)
        node_embed = torch.cat([node_embed, self.mask_token.to(device)], dim=0)
        roads = node_embed[road_nodes_tensor.to(device)]

        if self.config['use_temporal_encoding'] and timestamps is not None:
            print("Using Temporal Encoding")
            roads += self.temporal_encoding(timestamps.to(device))
        if self.config['use_time_embeddings'] and hour is not None and weekday is not None:
            print("Using Time Embedding")
            roads += self.hour_embed(hour.to(device))
            roads += self.weekday_embed(weekday.to(device))

        roads = self.pos_encoder(roads)

        if self.config['use_spatial_fusion'] and spatial_grid is not None:
            print("Using Spatia Fusion")
            spatial_features = self.cnn(spatial_grid)
            spatial_tokens = spatial_features.flatten(2).transpose(1, 2)
            spatial_repr = self.spatial_transformer(spatial_tokens)
            roads = self.cross_attn(roads, spatial_repr.expand(roads.size(0), -1, -1), spatial_repr.expand(roads.size(0), -1, -1))

        return self.traj_encoder(roads).mean(dim=1)


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

    def forward(self, batch, spatial_grid):
        device = next(self.parameters()).device
        B, L = batch['road_nodes'].shape

        node_embeddings = self.encode_graph(spatial_grid)
        node_embeddings = torch.cat([node_embeddings, self.mask_token.to(device)], dim=0)

        roads = node_embeddings[batch['road_nodes']]

        if self.temporal_encoding:
            roads += self.temporal_encoding(batch['timestamps'].to(device))
        if self.hour_embed:
            roads += self.hour_embed(batch['hour'].to(device))
        if self.weekday_embed:
            roads += self.weekday_embed(batch['weekday'].to(device))

        roads = self.pos_encoder(roads)

        if self.config['use_spatial_fusion']:
            spatial_features = self.cnn(spatial_grid.to(device))
            spatial_tokens = spatial_features.flatten(2).transpose(1, 2)
            spatial_repr = self.spatial_transformer(spatial_tokens).expand(B, -1, -1)
            roads = self.cross_attn(roads, spatial_repr, spatial_repr)

        fused = self.traj_encoder(roads)

        # MTM
        mtm_x = fused[batch['mtm_mask']]
        L_mtm = self.mtm(mtm_x, origin_nodes=batch['mtm_labels'])

        # Contrastive Losses
        L_cl_traj = torch.tensor(0.0, device=device)
        L_cl_node = torch.tensor(0.0, device=device)
        L_cl_node_node = torch.tensor(0.0, device=device)

        if self.config['use_traj_traj_cl']:
            traj_embed = self.projector(fused.mean(dim=1))
            road_nodes_aug = batch['road_nodes'].clone()
            for i in range(B):
                num_mask = max(1, int(0.15 * L))
                mask_indices = torch.randperm(L)[:num_mask]
                road_nodes_aug[i, mask_indices] = self._x.shape[0]
            traj_embed_aug = self.projector(self.encode_sequence(
                road_nodes_aug, batch['timestamps'], batch['hour'], batch['weekday'], spatial_grid
            ))
            L_cl_traj = contrastive_loss(traj_embed, traj_embed_aug, loss_type=self.config['contrastive_type'])

        if self.config['use_traj_node_cl']:
            traj_embed = self.projector(fused.mean(dim=1))
            L_cl_node = self.node_sequence_contrastive_loss(traj_embed, batch['road_nodes'], node_embeddings)

        if self.config['use_node_node_cl']:
            L_cl_node_node = self.node_node_contrastive_loss(node_embeddings[:-1])

        return L_mtm, L_cl_traj, L_cl_node, L_cl_node_node
