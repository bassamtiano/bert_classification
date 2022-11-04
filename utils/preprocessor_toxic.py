import re

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import pandas as pd

import pytorch_lightning as pl

from transformers import BertTokenizer

class PreprocessorToxic(pl.LightningDataModule):

    def __init__(self,  max_length = 100, batch_size = 40):
        super(PreprocessorToxic, self).__init__()
        self.tokenizers = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')
        self.max_length = max_length
        self.batch_size = batch_size

    def clean_str(self, string):
        string = string.lower()
        string = re.sub(r"[^A-Za-z0-9(),!?\'\-`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\n", "", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        string = string.strip()

        return string

    def load_data(self):
        data = pd.read_csv("data/multilabel/preprocessed_indonesian_toxic_tweet.csv")
        data = data.dropna(how="any")

        condition_empty_label = data[
            (
                (data['HS'] == 0) &
                (data['Abusive'] == 0) &
                (data['HS_Individual'] == 0) &
                (data['HS_Group'] == 0) &
                (data['HS_Religion'] == 0) &
                (data['HS_Race'] == 0) &
                (data['HS_Physical'] == 0) &
                (data['HS_Gender'] == 0) &
                (data['HS_Other'] == 0) &
                (data['HS_Weak'] == 0) &
                (data['HS_Moderate'] == 0) &
                (data['HS_Strong'] == 0)
            )
        ].index

        data = data.drop(condition_empty_label)

        tweet = data["Tweet"].apply(lambda x: self.clean_str(x))
        tweet = tweet.values.tolist()
        
        label = data.drop(["Tweet"], axis = 1)
        label = label.values.tolist()
        # Nama kolom-kolom di dataset
        self.labels = data.columns.tolist()[1:]

        print(self.labels)

        x_input_ids, x_token_type_ids, x_attention_mask = [], [], []

        for tw in tweet:
            tkn_tweet = self.tokenizers(text = tw,
                                        max_length = self.max_length,
                                        padding = 'max_length',
                                        truncation = True)
            
            x_input_ids.append(tkn_tweet['input_ids'])
            x_token_type_ids.append(tkn_tweet['token_type_ids'])
            x_attention_mask.append(tkn_tweet['attention_mask'])
        
        x_input_ids = torch.tensor(x_input_ids)
        x_token_type_ids = torch.tensor(x_token_type_ids)
        x_attention_mask = torch.tensor(x_attention_mask)
        y = torch.tensor(label)

        tensor_dataset = TensorDataset(x_input_ids, x_token_type_ids, x_attention_mask, y)

        train_valid_dataset, test_dataset = torch.utils.data.random_split(
            tensor_dataset, [
                round(len(tensor_dataset) * 0.8),
                len(tensor_dataset) - round(len(tensor_dataset) * 0.8)
            ]
        )

        train_len = round(len(train_valid_dataset) * 0.9)
        valid_len = len(train_valid_dataset) - round(len(train_valid_dataset) * 0.9)

        train_dataset, valid_dataset = torch.utils.data.random_split(
            tensor_dataset, [
                train_len, valid_len
            ]
        )

        return train_dataset, valid_dataset, test_dataset

    def get_labels(self):
        return self.labels
    
    def setup(self, stage = None):
        train_data, valid_data, test_data = self.load_data()
        if stage == "fit":
            self.train_data = train_data
            self.valid_data = valid_data
        elif stage == "predict":
            self.test_data = test_data

    def train_dataloader(self):
        sampler = RandomSampler(self.train_data)
        return DataLoader(
            dataset = self.train_data,
            batch_size = self.batch_size,
            sampler = sampler,
            num_workers = 3
        )

    def val_dataloader(self):
        sampler = RandomSampler(self.valid_data)
        return DataLoader(
            dataset = self.valid_data,
            batch_size = self.batch_size,
            sampler = sampler,
            num_workers = 3
        )

    def predict_dataloader(self):
        sampler = SequentialSampler(self.test_data)
        return DataLoader(
            dataset = self.test_data,
            batch_size = self.batch_size,
            sampler = sampler,
            num_workers = 3
        )
