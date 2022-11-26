import re
import pandas as pd

import torch

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer

class PreprocessorHierarcy():
    def __init__(self, 
                 max_length,
                 dir_dataset,
                 dir_tree, ) -> None:
        super(PreprocessorHierarcy, self).__init__

        self.dir_dataset = dir_dataset
        self.dir_tree = dir_tree

        self.tokenizers = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')
        self.max_length = max_length

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

    def load_tree(self):
        tree_level = {}
        level_tree = {}
        with open(self.dir_tree, "r") as dr:
            for line in dr:
                line = line[:-1].lower()
                category = line.split(" > ")[-1]
                level = len(line.split(" > "))
                if category not in tree_level:
                    tree_level[category] = level
                    if level not in level_tree:
                        level_tree[level] = []
                    level_tree[level] += [category]
        return tree_level, level_tree
    
    def encode_text(self, text):
        tkn = self.tokenizers(text = text,
                                 max_length = self.max_length,
                                 padding = "max_length",
                                 truncation = True)

        return tkn['input_ids'], tkn['token_type_ids'], tkn['attention_mask']

    def load_data(self):
        tree_level, level_tree = self.load_tree()

        data = pd.read_csv(self.dir_dataset)
        x_input_ids, x_token_type_ids, x_attention_mask, y, y_flat = [], [], [], [], []

        # Flat no label = 96

        for i, line in enumerate(data.values.tolist()):
            input_ids, token_type_ids, attention_mask = self.encode_text(line[0])
            
            flat_label = line[3].split(" > ")[-1]
            flat_binary = [0]*len(level_tree[3])
            flat_binary[level_tree[3].index(flat_label.lower())] = 1
            

            kategori = line[3]
            hierarcy = [set([tree_level[cat.lower()]]) for cat in kategori.split(" > ")]

            # Binarizer of hierarcy
            hierarcy_binary = []
            for level, cat in enumerate(kategori.split(" > "), 1):
                binary = [0]*len(level_tree[level])
                binary[level_tree[level].index(cat.lower())] = 1
                hierarcy_binary.append(binary)

            y.append(hierarcy_binary)
            y_flat.append(flat_binary)

            x_input_ids.append(input_ids)
            x_token_type_ids.append(token_type_ids)
            x_attention_mask.append(attention_mask)

            y.append(hierarcy)

        x_input_ids = torch.tensor(x_input_ids)
        x_token_type_ids = torch.tensor(x_token_type_ids)
        x_attention_mask = torch.tensor(x_attention_mask)
        y = torch.tensor(y)
        y_flat = torch.tensor(y_flat)

        tensor_dataset = TensorDataset(x_input_ids, x_token_type_ids, x_attention_mask, y, y_flat)

        train_valid_dataset, test_dataset = torch.utils.data.random_split(
            tensor_dataset, [
                round(len(tensor_dataset) * 0.8),
                len(tensor_dataset) - round(len(tensor_dataset) * 0.8)
            ]
        )

        train_len = round(len(train_valid_dataset) * 0.9)
        valid_len = len(train_valid_dataset) - round(len(train_valid_dataset) * 0.9)

        train_dataset, valid_dataset = torch.utils.data.random_split(
            train_valid_dataset, [
                train_len, valid_len
            ]
        )

        return train_dataset, valid_dataset, test_dataset



    def preprocessor(self):
        # tree_level, level_tree = self.load_tree()
        # print(level_tree)
        # print(tree_level["elektronik"])
        self.load_data()