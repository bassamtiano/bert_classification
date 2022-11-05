import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from transformers import BertModel

from torchmetrics import AUROC

class MultiLabelModel(pl.LightningModule):
    def __init__(self,
                 labels,
                 lr = 1e-4,
                 embedding_dim = 768,
                 in_channels = 8,
                 out_channels = 40,
                 num_classes = 12,
                 dropout = 0.3) -> None:
        super(MultiLabelModel, self).__init__()

        self.lr = lr
        self.labels = labels

        torch.manual_seed(1)
        random.seed(1)

        self.criterion = nn.BCELoss()

        ks = 3
        self.bert = BertModel.from_pretrained('indolem/indobert-base-uncased', output_hidden_states = True)

        self.pre_classifier = nn.Linear(768, 768)

        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, embedding_dim), groups=4)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (4, embedding_dim), groups=4)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (5, embedding_dim), groups=4)


        self.dropout = nn.Dropout(dropout)
        self.l1 = nn.Linear(ks * out_channels, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, token_type_ids, attention_mask):

        bert_out = self.bert(input_ids = input_ids, 
                             token_type_ids = token_type_ids, 
                             attention_mask = attention_mask)

        hidden_state = bert_out[2]

        hidden_state = torch.stack(hidden_state, dim = 1)
        hidden_state = hidden_state[:, -8:]
        # hidden state shape = batch size, 8 layer terakhir, max_length, embedding dim bert

        x = [
            F.relu(self.conv1(hidden_state).squeeze(3)),
            F.relu(self.conv2(hidden_state).squeeze(3)),
            F.relu(self.conv3(hidden_state).squeeze(3))
        ]

        x = [
            F.max_pool1d(i, i.size(2)).squeeze(2) for i in x
        ]

        x = torch.cat(x, dim = 1) # batch size * output cnn yang di concat

        
        x = self.dropout(x)

        logits = self.l1(x)
        logits = self.sigmoid(logits)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = batch

        out = self(input_ids = x_input_ids,
                   token_type_ids = x_token_type_ids,
                   attention_mask = x_attention_mask)
        # Shape outputnya (batch no, num_labels)
        loss = self.criterion(out.cpu(), y.float().cpu())

        self.log("train_loss", loss)

        return {"loss": loss, "predictions": out, "labels": y}
        
    def validation_step(self, batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = batch

        out = self(input_ids = x_input_ids,
                   token_type_ids = x_token_type_ids,
                   attention_mask = x_attention_mask)

        loss = self.criterion(out.cpu(), y.float().cpu())

        self.log("val_loss", loss)

        return {"val_loss": loss}

    def predict_step(self, batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = batch

        out = self(input_ids = x_input_ids,
                   token_type_ids = x_token_type_ids,
                   attention_mask = x_attention_mask)
        # loss = self.criterion(out.cpu(), y.float().cpu())

        return out

    def training_epoch_end(self, outputs):
        labels = []
        predictions = []
        for outputs in outputs:
            for out_lbl in outputs["labels"].detach().cpu():
                labels.append(out_lbl)
            for out_pred in outputs["predictions"].detach().cpu():
                predictions.append(out_pred)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)


        for i, name in enumerate(self.labels):
            auroc = AUROC(num_classes = len(self.labels))
            class_roc_auc = auroc(predictions[:, i], labels[:, i])

            print(f"{name} \t : {class_roc_auc}")

            self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)