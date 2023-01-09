import random
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from sklearn.metrics import classification_report

class HierarchyModel(pl.LightningModule):
    def __init__(self, 
                 lr, 
                 dropout,
                 in_channels,
                 out_channels,
                 n_out,
                 embedding_dim) -> None:
        super(HierarchyModel, self).__init__()

        pl.seed_everything(1)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, embedding_dim), groups=4)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (4, embedding_dim), groups=4)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (5, embedding_dim), groups=4)

        self.lr = lr
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, n_out)

    def forward(self,):
        

    def configure_optimizers(self):
        optimizers = torch.optim.Adam(self.parameters(), lr = self.lr)
        return optimizers

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def prediction_step(self, batch, batch_idx):
        pass

