import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import math

# === Positional Encoding ===
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, D]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


# === GAT Model ===
class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=0.2)
        self.gat2 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=0.2)

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x

# === CNN Module ===
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

# === Spatial Transformer ===
class SpatialTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer(x)

# === Cross-Attention Block ===    
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, query, key, value):
        attn_output, _ = self.attn(query, key, value)
        return attn_output  # [B, L, D]


# === Trajectory Encoder ===
class TrajectoryEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoding = PositionalEncoding(embed_dim)

    def forward(self, x):
        return self.transformer(self.pos_encoding(x))
    
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

# === Trajectory Model ===
class TrajectoryModel(nn.Module):
    def __init__(self, config, graph_data):
        super().__init__()
        self.gat = GATModel(config['node_feat_dim'], config['gat_hidden'], config['embed_dim'])
        self.cnn = CNNModel(config['spatial_in_channels'], config['embed_dim'])
        self.spatial_transformer = SpatialTransformer(config['embed_dim'], config['num_heads'], config['spatial_layers'])

        self.pos_encoder = PositionalEncoding(config['embed_dim'])
        encoder_layer = nn.TransformerEncoderLayer(config['embed_dim'], config['num_heads'], batch_first=True)
        self.traj_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['traj_layers'])

        self.cross_attn = CrossAttention(config['embed_dim'], config['num_heads'])
        self.mask_token = nn.Parameter(torch.zeros(1, config['embed_dim']))

        self.mtm = MTMHead(config['embed_dim'], config['num_nodes'])

        self._x, self._edge_index = graph_data

    def encode_graph(self, spatial_grid=None):
        """
        Returns: [num_nodes, embed_dim]
        """
        self.eval()
        with torch.no_grad():
            x = self._x.to(next(self.parameters()).device)
            edge_index = self._edge_index.to(next(self.parameters()).device)
            node_embeddings = self.gat(x, edge_index)

            if spatial_grid is not None:
                spatial_features = self.cnn(spatial_grid)  # [1, D, H, W]
                spatial_tokens = spatial_features.flatten(2).transpose(1, 2)  # [1, S, D]
                spatial_repr = self.spatial_transformer(spatial_tokens)  # [1, S, D]
                context_vector = spatial_repr.mean(dim=1)  # [1, D]
                node_embeddings = node_embeddings + context_vector  # broadcast

            return node_embeddings

    def encode_sequence(self, road_nodes_tensor, spatial_grid=None):
        """
        road_nodes_tensor: [B, L] — each row is a trajectory of node IDs
        Returns: [B, D] — trajectory-level embeddings
        """
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            node_embeddings = self.encode_graph(spatial_grid)  # already fused if needed

            # Add [MASK] token to end
            node_embeddings = torch.cat([node_embeddings, self.mask_token.to(device)], dim=0)
            road_embed = node_embeddings[road_nodes_tensor.to(device)]  # [B, L, D]

            # Positional encoding
            road_embed = self.pos_encoder(road_embed)

            # Cross-attn with spatial grid if provided
            if spatial_grid is not None:
                spatial_features = self.cnn(spatial_grid)  # [1, D, H, W]
                spatial_tokens = spatial_features.flatten(2).transpose(1, 2)  # [1, S, D]
                spatial_repr = self.spatial_transformer(spatial_tokens)  # [1, S, D]
                road_embed = self.cross_attn(road_embed, spatial_repr.expand(road_embed.size(0), -1, -1), spatial_repr.expand(road_embed.size(0), -1, -1))

            # Transformer encoding
            traj_encoded = self.traj_encoder(road_embed)  # [B, L, D]
            traj_embed = traj_encoded.mean(dim=1)  # mean-pool over tokens

            return traj_embed  # [B, D]
        
    def forward(self, batch, spatial_grid):
        device = next(self.parameters()).device
        B, L = batch['road_nodes'].shape

        node_embeddings = self.gat(self._x.to(device), self._edge_index.to(device))
        node_embeddings = torch.cat([node_embeddings, self.mask_token.to(device)], dim=0)

        road_embeddings = node_embeddings[batch['road_nodes'].to(device)]  # [B, L, D]
        spatial_features = self.cnn(spatial_grid.to(device))  # [1, D, H, W]
        spatial_tokens = spatial_features.flatten(2).transpose(1, 2)  # [1, S, D]
        spatial_repr = self.spatial_transformer(spatial_tokens).expand(B, -1, -1)  # [B, S, D]

        fused = self.cross_attn(road_embeddings, spatial_repr, spatial_repr)  # [B, L, D]
        fused = self.traj_encoder(fused)  # [B, L, D]

        mtm_x = fused[batch['mtm_mask']]
        loss = self.mtm(mtm_x, origin_nodes=batch['mtm_labels'])
        return loss


