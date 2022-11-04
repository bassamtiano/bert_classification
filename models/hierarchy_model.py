import random
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from sklearn.metrics import classification_report

class HierarchyModel(pl.LightningModule):
    def __init__(self) -> None:
        super(HierarchyModel).__init__()

    def forward(self,):
        pass

    def configure_optimizers(self):
        pass

    def training_step(self, batch, batch_idx):
        pass

