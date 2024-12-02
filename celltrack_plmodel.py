# celltrack_plmodel.py

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from celltrack_model import CellTrack_Model

class CellTrackLitModel(LightningModule):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=32, lr=0.001):
        super(CellTrackLitModel, self).__init__()
        self.model = CellTrack_Model(input_dim, hidden_dim, output_dim)
        self.lr = lr

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        # Forward pass and calculate loss
        predictions = self(batch)
        loss = F.binary_cross_entropy(predictions, batch.y.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # Forward pass and calculate loss
        predictions = self(batch)
        loss = F.binary_cross_entropy(predictions, batch.y.float())
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        # Use Adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
