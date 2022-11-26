import random
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from sklearn.metrics import classification_report

class HierarchyModel(pl.LightningModule):
    def __init__(self, lr) -> None:
        super(HierarchyModel).__init__()
        self.lr = lr

    def forward(self,):
        pass

    def configure_optimizers(self):
        optimizers = torch.optim.Adam(self.parameters(), lr = self.lr)
        return optimizers

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def prediction_step(self, batch, batch_idx):
        pass

