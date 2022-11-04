import sys
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from sklearn.metrics import classification_report

from transformers import BertModel

from torchmetrics import AUROC

class MultiLabelModel(pl.LightningModule):
    def __init__(self,  
                 labels, 
                 lr = 1e-4,
                 embedding_dim = 768,
                 in_channels = 8, 
                 out_channels = 32,
                 num_classes = 12,
                 kernel_size = 10,
                 dropout = 0.3
                 ) -> None:
        super(MultiLabelModel, self).__init__()

        self.lr = lr
        self.labels = labels

        torch.manual_seed(1)
        random.seed(43)

        self.criterion = nn.BCELoss()

        ks = 3

        self.bert_model = BertModel.from_pretrained('indolem/indobert-base-uncased', output_hidden_states = True)
        self.pre_classifier = nn.Linear(768, 768)

        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, embedding_dim), padding=(2, 0), groups=4)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (4, embedding_dim), padding=(3, 0), groups=4)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (5, embedding_dim), padding=(4, 0), groups=4)

        # apply dropout
        self.dropout = nn.Dropout(dropout)
        self.l1 = nn.Linear(ks * out_channels, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_out = self.bert_model(input_ids = input_ids, 
                                   attention_mask = attention_mask, 
                                   token_type_ids = token_type_ids)
        hidden_state = bert_out[2]
        
        hidden_state = torch.stack(hidden_state, dim = 1)
        hidden_state = hidden_state[:, -8:]

        x = [
            F.relu(self.conv1(hidden_state).squeeze(3)),
            F.relu(self.conv2(hidden_state).squeeze(3)),
            F.relu(self.conv3(hidden_state).squeeze(3))
        ]

        # print(x.shape)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)

        x = self.dropout(x)
        logit = self.l1(x)
        logit = self.sigmoid(logit)
        return logit

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = train_batch

        out = self(input_ids = x_input_ids,
                   token_type_ids = x_token_type_ids,
                   attention_mask = x_attention_mask)

        loss = self.criterion(out, y.float())

        self.log("train_loss", loss)
        
        return {"loss": loss, "predictions": out, "labels": y}
        
    def validation_step(self, valid_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = valid_batch

        out = self(input_ids = x_input_ids,
                   token_type_ids = x_token_type_ids,
                   attention_mask = x_attention_mask)
        
        loss = self.criterion(out.cpu(), y.float().cpu())

        self.log("val_loss", loss)

        return loss

    def predict_step(self, test_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = test_batch
        out = self(input_ids = x_input_ids,
                   token_type_ids = x_token_type_ids,
                   attention_mask = x_attention_mask)
        
        loss = self.criterion(out.cpu(), y.float().cpu())

        return out


    def training_epoch_end(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        # results = []

        for i, name in enumerate(self.labels):
            auroc = AUROC(num_classes=len(self.labels))
            class_roc_auc = auroc(predictions[:, i], labels[:, i])
            # results.append(class_roc_auc)
            print(f"{name} \t: {class_roc_auc}")

            self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)

        # print(results)
