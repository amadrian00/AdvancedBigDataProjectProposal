import torch
from torch_geometric.nn import GCNConv, global_add_pool

class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        hidden = 64
        self.embedding = torch.nn.Linear(3, 1)
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = torch.nn.Linear(hidden, out_channels)


    def forward(self, x, edge_index, edge_attr, batch):
        edge_attr = self.embedding(edge_attr).relu()
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = global_add_pool(x, batch)
        return self.lin(x).squeeze(1)