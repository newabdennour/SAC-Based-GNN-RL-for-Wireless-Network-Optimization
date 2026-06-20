import torch
import torch.nn as nn
import torch.nn.functional as F

class SageLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.self_proj = nn.Linear(in_dim, out_dim)
        self.neigh_proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        h = self.self_proj(x) + self.neigh_proj(torch.sparse.mm(adj, x))
        return self.dropout(F.gelu(self.norm(h)))

class ResidualGraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, depth=3, dropout=0.25):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([SageLayer(hidden_dim, hidden_dim, dropout) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x, adj):
        h = F.gelu(self.input(x))
        for layer in self.layers:
            h = h + layer(h, adj)
        return self.head(h)
