import sys
import re
import pandas as pd

import pickle

import torch

import os

from tqdm import tqdm

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer

class PreprocessorHierarcy():
    def __init__(self, 
                 max_length,
                 dir_dataset,
                 train_dataset_dir,
                 test_dataset_dir,
                 dir_tree, ) -> None:
        super(PreprocessorHierarcy, self).__init__

        self.dir_dataset = dir_dataset
        self.dir_tree = dir_tree

        self.train_dataset_dir = train_dataset_dir
        self.test_dataset_dir = test_dataset_dir

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

        parent2child = {}

        with open(self.dir_tree, "r") as dr:
            for i, line in tqdm(enumerate(dr)):
                line = line[:-1].lower()
                category = line.split(" > ")[-1]
                category_all = line.split(" > ")


                # if len(category_all) > 1:
                #     for i_cat, cat in enumerate(category_all):
                #         if i_cat > 0:
                #             if category_all[i_cat - 1] not in parent2child:
                #                 parent2child[category_all[i_cat - 1]] = set()
                #             else:
                #                 parent2child[category_all[i_cat - 1]].add(cat)

                for i, cat in enumerate(category_all):
                    if i > 0:
                        parent = category_all[i - 1]
                        try:
                            parent2child[parent].add(cat)
                        except:
                            parent2child[parent] = set()
                            parent2child[parent].add(cat)
                            
                
                
                level = len(line.split(" > "))
                if category not in tree_level:
                    tree_level[category] = level
                    if level not in level_tree:
                        level_tree[level] = []
                    level_tree[level] += [category]

        return tree_level, level_tree, parent2child
    
    def encode_text(self, text):
        tkn = self.tokenizers(text = text,
                                 max_length = self.max_length,
                                 padding = "max_length",
                                 truncation = True)

        return tkn['input_ids'], tkn['token_type_ids'], tkn['attention_mask']

    def split_dataset(self):
        if os.path.exists(self.train_dataset_dir) and os.path.exists(self.test_dataset_dir):
            print("load split csv dataset")
            train_data = pd.read_csv(self.train_dataset_dir)
            test_data = pd.read_csv(self.test_dataset_dir)
        else: 
            print("create split csv dataset")
            data = pd.read_csv(self.dir_dataset)
            data = data.sample(frac = 1)

            data_len = data.shape[0]
            train_len : int = int(data_len * 0.7)

            train_data = data.iloc[:train_len, :]
            test_data = data.iloc[train_len:, :]
            
            # Cek Label balance / tidak
            print(len(train_data['leaf'].value_counts()))
            print("=" * 30)
            print(len(test_data['leaf'].value_counts()))
            
            train_data = train_data.to_csv(self.train_dataset_dir, index = False)
            test_data = test_data.to_csv(self.test_dataset_dir, index = False)

        return train_data, test_data

    def preprocess_test_dataset(self, test_dataset):
        # tree leve untuk mengetahui label label yang ada di level tertentu
        # level tree untuk mengetahui level dari label
        tree_level, level_tree, parent2child = self.load_tree()

        x_input_ids, y_flat = [], []

        # Flat no label = 96
        parentid = {p: i for i, p in enumerate(parent2child.keys())}
        overall_dataset = [ [] for i in range(len(parentid))]
        
        # Hierarcy vs flat
        for i, line in tqdm(enumerate(test_dataset.values.tolist())):
            input_ids, token_type_ids, attention_mask = self.encode_text(line[0])
            

            flat_label = line[3].split(" > ")[-1]

            flat_binary = [0] * len(level_tree[3])
            flat_binary[level_tree[3].index(flat_label.lower())] = 1

            # Ingin tahu level ke berapa dari si label
            kategori = line[3].split(" > ")
            # hierarcy = [set([tree_level[cat.lower()]]) for cat in kategori.split(" > ")]

            # Pemisahan hierarcy data
            for i_f, cat in enumerate(kategori[:-1]):
                
                child = kategori[i_f + 1].lower()
                member = parent2child[cat.lower()]
                member = list(member)

                i_child = member.index(child)
                binary_member = [0] * len(member)
                binary_member[i_child] = 1
                
                # Buat ambil id parent di overall dataset
                i_parent = parentid[cat.lower()]

                if "input_ids" not in overall_dataset[i_parent]:
                    print("inisialisasi")
                    overall_dataset[i_parent] = {"input_ids" : [], "y": []}
                
                overall_dataset[i_parent]["input_ids"].append(input_ids)
                overall_dataset[i_parent]["y"].append(binary_member)
                

            # if i > 10 :
            #     break
           

            y_flat.append(flat_binary)
            x_input_ids.append(input_ids)


    def preprocess_train_dataset(self, train_dataset):
        # tree leve untuk mengetahui label label yang ada di level tertentu
        # level tree untuk mengetahui level dari label
        tree_level, level_tree, parent2child = self.load_tree()

        x_input_ids, y_flat = [], []

        # Flat no label = 96
        parentid = {p: i for i, p in enumerate(parent2child.keys())}
        overall_dataset = [ [] for i in range(len(parentid))]
        
        # Hierarcy vs flat
        for i, line in tqdm(enumerate(train_dataset.values.tolist())):
            input_ids, token_type_ids, attention_mask = self.encode_text(line[0])
            
            flat_label = line[3].split(" > ")[-1]

            flat_binary = [0] * len(level_tree[3])
            flat_binary[level_tree[3].index(flat_label.lower())] = 1

            # Ingin tahu level ke berapa dari si label
            kategori = line[3].split(" > ")
            # hierarcy = [set([tree_level[cat.lower()]]) for cat in kategori.split(" > ")]

            # Pemisahan hierarcy data
            for i_f, cat in enumerate(kategori[:-1]):
                
                child = kategori[i_f + 1].lower()
                member = parent2child[cat.lower()]
                member = list(member)

                i_child = member.index(child)
                binary_member = [0] * len(member)
                binary_member[i_child] = 1
                
                # Buat ambil id parent di overall dataset
                i_parent = parentid[cat.lower()]

                if "input_ids" not in overall_dataset[i_parent]:
                    print("inisialisasi")
                    overall_dataset[i_parent] = {"input_ids" : [], "y": []}
                
                overall_dataset[i_parent]["input_ids"].append(input_ids)
                overall_dataset[i_parent]["y"].append(binary_member)
                

            # if i > 10 :
            #     break
           

            y_flat.append(flat_binary)
            x_input_ids.append(input_ids)
        
        
        
        hierarcy_dataset = []
        for i, od in enumerate(overall_dataset):
            
            o_input_ids = od["input_ids"]
            o_y = od["y"]

            train_dataset, valid_dataset = self.splitting_data(o_input_ids, o_y)
            hierarcy_dataset.append([train_dataset, valid_dataset])
        # print(hierarcy_dataset)
        # sys.exit()
        flat_train_dataset, flat_valid_dataset = self.splitting_data(x_input_ids, y_flat)
        flat_dataset = [flat_train_dataset, flat_valid_dataset]

        datasets = {"flat": flat_dataset, "hierarcy": hierarcy_dataset}
        print("saving preprocessed dataset")
        with open("data/hierarcy/preprocessed/preprocessed_all.pkl", "wb") as wh:
            pickle.dump(datasets, wh, protocol=pickle.HIGHEST_PROTOCOL)

        return datasets

    def load_data(self):
        train_csv_data, test_csv_data = self.split_dataset()

        train_datasets = self.preprocess_train_dataset(train_csv_data)
        test_datasets = self.preprocess_test_dataset(test_csv_data)
        return train_datasets, test_datasets
        
    def splitting_data(self, x_input_ids, y):
        x_input_ids = torch.tensor(x_input_ids)
        y = torch.tensor(y)

        tensor_dataset = TensorDataset(x_input_ids, y)

        # train_valid_dataset, test_dataset = torch.utils.data.random_split(
        #     tensor_dataset, [
        #         round(len(tensor_dataset) * 0.8),
        #         len(tensor_dataset) - round(len(tensor_dataset) * 0.8)
        #     ]
        # )

        train_len = round(len(tensor_dataset) * 0.9)
        valid_len = len(tensor_dataset) - train_len

        train_dataset, valid_dataset = torch.utils.data.random_split(
            tensor_dataset, [
                train_len, valid_len
            ]
        )

        return train_dataset, valid_dataset

    def preprocessor(self):
        if not os.path.exists("data/hierarcy/preprocessed/preprocessed_all.pkl"):
            print("Preprocessing dataset")
            train_datasets, test_datasets = self.load_data()
        else:
            print("Load preprocessed dataset")
            with open("data/hierarcy/preprocessed/preprocessed_all.pkl", "rb") as f:
                train_datasets, test_datasets = pickle.load(f)
                print(len(train_datasets["hierarcy"]))
                print(len(train_datasets["hierarcy"]))
                print(len(train_datasets["hierarcy"][1]))
                print(train_datasets["hierarcy"][1])


