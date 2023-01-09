import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel

class MultiClassOnly(nn.Module):
    def __init__(self, n_out, dropout):
        super(MultiClassOnly, self).__init__()
        
        self.bert_model = BertModel.from_pretrained('indolem/indobert-base-uncased')
        self.pre_classifier = nn.Linear(768, 768)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, n_out)
        # self.tanh = nn.Tanh()

    def forward(self, input_ids, attention_mask, token_type_ids):
        
        
        bert_out = self.bert_model(
                        input_ids = input_ids
                   )
        
        hidden_state = bert_out[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output