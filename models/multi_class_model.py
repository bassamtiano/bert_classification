import sys
import random
import torch
import torch.nn as nn

import pytorch_lightning as pl

from transformers import BertModel

from sklearn.metrics import classification_report

from torchmetrics import AUROC

class MultiClassModel(pl.LightningModule):
    def __init__(self, n_out, dropout, lr):
        super(MultiClassModel, self).__init__()

        torch.manual_seed(1)
        random.seed(43)

        

        self.bert = BertModel.from_pretrained('indolem/indobert-base-uncased')
        self.pre_classifier = nn.Linear(768, 768)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, n_out)
        # self.tanh = nn.Tanh()

        self.lr = lr

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_out = self.bert(
                        input_ids = input_ids,
                        attention_mask = attention_mask,
                        token_type_ids = token_type_ids,
                   )
        hidden_state = bert_out[0]
        pooler = hidden_state[:, 0]
        # pooler = bert_out[1]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = train_batch
        # x input ids = dimensi batch x max_length

        out = self(input_ids = x_input_ids, 
                   attention_mask = x_attention_mask,
                   token_type_ids = x_token_type_ids)
        loss = self.criterion(out, target = y.float())

        pred = out.argmax(1).cpu()
        true = y.argmax(1).cpu()
        report = classification_report(true, pred, output_dict = True, zero_division = 0)

        self.log("accuracy", report["accuracy"], prog_bar = True)
        self.log("loss", loss)

        return loss

    def validation_step(self, valid_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = valid_batch

        out = self(input_ids = x_input_ids, 
                   attention_mask = x_attention_mask,
                   token_type_ids = x_token_type_ids)
        loss = self.criterion(out, target = y.float())

        pred = out.argmax(1).cpu()
        true = y.argmax(1).cpu()
        report = classification_report(true, pred, output_dict = True, zero_division = 0)

        self.log("accuracy", report["accuracy"], prog_bar = True)
        self.log("loss", loss)

        return loss


    def predict_step(self, test_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = test_batch

        out = self(input_ids = x_input_ids, 
                   attention_mask = x_attention_mask,
                   token_type_ids = x_token_type_ids)

        pred = out.argmax(1).cpu()
        true = y.argmax(1).cpu()
        
        return pred, true
    

    # # Tambahan

    # def training_epoch_end(self, outputs):
    #     labels = []
    #     predictions = []
    #     for output in outputs:
    #         for out_labels in output["labels"].detach().cpu():
    #             labels.append(out_labels)
    #         for out_predictions in output["predictions"].detach().cpu():
    #             predictions.append(out_predictions)

    #     labels = torch.stack(labels).int()
    #     predictions = torch.stack(predictions)

    #     # results = []

    #     for i, name in enumerate(self.labels):
    #         auroc = AUROC(num_classes=len(self.labels))
    #         class_roc_auc = auroc(predictions[:, i], labels[:, i])
    #         # results.append(class_roc_auc)
    #         print(f"{name} \t: {class_roc_auc}")

    #         self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)

    