import torch
from sac_graph5g.models.gnn import ResidualGraphSAGE, SageLayer

def test_sage_layer():
    in_dim, out_dim = 16, 16
    layer = SageLayer(in_dim, out_dim, dropout=0.0)
    
    x = torch.randn(10, in_dim)
    adj = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        values=torch.ones(4),
        size=(10, 10)
    )
    
    out = layer(x, adj)
    assert out.shape == (10, out_dim)

def test_residual_graphsage():
    in_dim, hidden_dim, out_dim = 16, 32, 3
    model = ResidualGraphSAGE(in_dim, hidden_dim, out_dim, depth=2)
    
    x = torch.randn(10, in_dim)
    adj = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        values=torch.ones(4),
        size=(10, 10)
    )
    
    out = model(x, adj)
    assert out.shape == (10, out_dim)
