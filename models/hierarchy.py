import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel

class Hierarchy(nn.Module):
    def __init__(self,
                 lr,
                 embedding_dim,
                 in_channels,
                 out_channels,
                 num_classes,
                 dropout) -> None:
        super(Hierarchy, self).__init__()

        self.lr = lr
        self.criterion = nn.BCELoss()

        ks = 3
        self.bert = BertModel.from_pretrained('indolem/indobert-base-uncased', output_hidden_states = True)

        self.pre_classifier = nn.Linear(768, 768)

        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, embedding_dim), groups=4)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (4, embedding_dim), groups=4)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (5, embedding_dim), groups=4)

        self.dropout = nn.Dropout(dropout)
        self.l1 = nn.Linear(ks * out_channels, num_classes)
        self.relu = nn.ReLU()

    def forward(self, input_ids):
        bert_out = self.bert(input_ids = input_ids)
        hidden_state = bert_out[2]

        hidden_state = torch.stack(hidden_state, dim = 1)
        hidden_state = hidden_state[:, -8:]

        x = [
            F.relu(self.conv1(hidden_state).squeeze(3)),
            F.relu(self.conv2(hidden_state).squeeze(3)),
            F.relu(self.conv3(hidden_state).squeeze(3))
        ]

        x = [
            F.max_pool1d(i, i.size(2)).squeeze(2) for i in x
        ]

        x = torch.cat(x, dim = 1)

        x = self.dropout(x)
        x = self.l1(x)

        x = torch.nn.Tanh()(x)

        return x