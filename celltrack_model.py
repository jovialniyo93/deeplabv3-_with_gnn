# celltrack_model.py

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, EdgeConv

class CellTrack_Model(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=32):
        super(CellTrack_Model, self).__init__()
        
        # Node feature processing
        self.node_conv1 = GCNConv(input_dim, hidden_dim)
        
        # Edge feature processing
        self.edge_conv = EdgeConv(nn=nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ))

        # Classifier for edge predictions
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        x = self.node_conv1(data.x, data.edge_index).relu()
        edge_features = self.edge_conv(x, data.edge_index)
        out = self.classifier(edge_features).squeeze(-1)
        return out
