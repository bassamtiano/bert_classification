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
                 batch_size,
                 dir_dataset,
                 train_dataset_dir,
                 test_dataset_dir,
                 preprocessed_dir,
                 tree_helper,
                 split_data_section) -> None:
        super(PreprocessorHierarcy, self).__init__

        self.max_length = max_length
        self.batch_size = batch_size

        self.dir_dataset = dir_dataset
        
        self.tree_helper = tree_helper

        self.train_dataset_dir = train_dataset_dir
        self.test_dataset_dir = test_dataset_dir
        self.preprocessed_dir = preprocessed_dir

        self.split_data_section = split_data_section

        self.tokenizers = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')
        

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

    def split_by_hierarchy_section(self, 
                                   kategori, 
                                   parent_id,
                                   parent2child,
                                   overall_dataset, 
                                   input_ids):
        for i_f, cat in enumerate(kategori[:-1]):
                
            child = kategori[i_f + 1].lower()
            member = parent2child[cat.lower()]
            member = list(member)

            i_child = member.index(child)
            binary_member = [0] * len(member)
            binary_member[i_child] = 1
            
            # Buat ambil id parent di overall dataset
            i_parent = parent_id[cat.lower()]

            if "input_ids" not in overall_dataset[i_parent]:
                print("inisialisasi")
                overall_dataset[i_parent] = {"input_ids" : [], "y": []}
            
            overall_dataset[i_parent]["input_ids"].append(input_ids)
            overall_dataset[i_parent]["y"].append(binary_member)

        return overall_dataset

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

    def preprocess_dataset(self, train_dataset, level, train_status = "train"):
        # tree leve untuk mengetahui label label yang ada di level tertentu
        # level tree untuk mengetahui level dari label
        tree_level, level_tree, parent2child, level_tree_item, level_tree_ids = self.tree_helper.load_tree()
        
        x_input_ids, y_all = [], []

        # Flat no label = 96
        parent_id = {p: i for i, p in enumerate(parent2child.keys())}
        overall_dataset = [ [] for i in range(len(parent_id))]
        
        # Hierarcy vs flat
        progress_preprocessor_train = tqdm(train_dataset.values.tolist())

        for i, line in enumerate(progress_preprocessor_train):
            input_ids, _, _ = self.encode_text(line[0])
            
            # hierarcy = [set([tree_level[cat.lower()]]) for cat in kategori.split(" > ")]

            # Pemisahan hierarcy data
            if level == "section":
                # Bagian Pemisahan per section of hierarcy

                # Ingin tahu level ke berapa dari si label
                kategori = line[3].split(" > ")

                overall_dataset = self.split_by_hierarchy_section(kategori = kategori,
                                                                  parent_id = parent_id,
                                                                  parent2child = parent2child,
                                                                  overall_dataset = overall_dataset,
                                                                  input_ids = input_ids)
            elif level == "flat":
                flat_label = line[3].split(" > ")[-1]

                flat_binary = [0] * len(level_tree[3])
                flat_binary[level_tree[3].index(flat_label.lower())] = 1

                y_all.append(flat_binary)
            
            else :
                # Split based on section
                kat_level = line[3].split(" > ")[level - 1].lower()
                ids = level_tree_ids[level]
                
                y_ids = ids[kat_level]
                y = [0] * len(ids)
                y[y_ids] = 1

                y_all.append(y)


            x_input_ids.append(input_ids)
        
        file_name = ""

        if level == "section":
            file_name = "preprocessed_training_section.pkl"

            dataset = []
            for i, od in enumerate(overall_dataset):
                
                o_input_ids = od["input_ids"]
                o_y = od["y"]


                train_dataset, valid_dataset = self.splitting_data(o_input_ids, o_y)
                dataset.append([train_dataset, valid_dataset])
        
        elif level == "flat":
            if train_status == "train":
                file_name = "preprocessed_train_flat.pkl"
                flat_train_dataset, flat_valid_dataset = self.splitting_data(x_input_ids, y_all)
                dataset = [flat_train_dataset, flat_valid_dataset]
            else:
                file_name = "preprocessed_test_flat.pkl"
                x_input_ids = torch.tensor(x_input_ids)
                y_all = torch.tensor(y_all)

                dataset = TensorDataset(x_input_ids, y_all)


        else:
            if train_status == "train":
                file_name = "preprocessed_train_" + str(level) + ".pkl"
                level_train_dataset, level_valid_dataset = self.splitting_data(x_input_ids, y_all)
                dataset = [level_train_dataset, level_valid_dataset]
            else:
                file_name = "preprocessed_test_" + str(level) + ".pkl"
                x_input_ids = torch.tensor(x_input_ids)
                y_all = torch.tensor(y_all)

                dataset = TensorDataset(x_input_ids, y_all)


        with open(self.preprocessed_dir + file_name, "wb") as wh:
            pickle.dump(dataset, wh, protocol=pickle.HIGHEST_PROTOCOL)

        return dataset

    def preprocessor(self, level, train_status):
        
        # if os.path.exists(self.preprocessed_dir + "preprocessed_" + train_status + "_" + str(level) + ".pkl"):
        #     if train_status == "train":
        #         print("load preprocessed dataset " + str(level))
        #         with open(self.preprocessed_dir + "preprocessed_" + train_status + "_" + str(level) + ".pkl", "rb") as r_h:
        #             train_datasets, val_dataset = pickle.load(r_h)
        #     else:
        #         with open(self.preprocessed_dir + "preprocessed_" + train_status + "_" + str(level) + ".pkl", "rb") as r_h:
        #             test_datasets = pickle.load(r_h)
        # else:
        #     print("create preprocessed dataset " + str(level))
        #     train_csv_data, test_csv_data = self.split_dataset()
        #     if train_status == "train":
        #         train_datasets, val_dataset = self.preprocess_dataset(train_csv_data, level, train_status = train_status)
        #     else:
        #         test_datasets = self.preprocess_dataset(test_csv_data, level, train_status = train_status)

        

        if train_status == "train":
            if os.path.exists(self.preprocessed_dir + "preprocessed_" + train_status + "_" + str(level) + ".pkl"):
                print("load preprocessed dataset " + train_status + " " + str(level))
                with open(self.preprocessed_dir + "preprocessed_" + train_status + "_" + str(level) + ".pkl", "rb") as r_h:
                    train_datasets, val_datasets = pickle.load(r_h)
            else:
                print("create preprocessed dataset " + train_status + " " + str(level))
                train_csv_data, _ = self.split_dataset()
                train_datasets, val_datasets = self.preprocess_dataset(train_csv_data, level, train_status = train_status)
            
            train_sampler = RandomSampler(train_datasets)
            val_sampler = RandomSampler(val_datasets)

            train_dataloader = DataLoader(dataset = train_datasets,
                                          batch_size = self.batch_size,
                                          sampler = train_sampler,
                                          num_workers = 3)

            val_dataloader = DataLoader(dataset = val_datasets,
                                        batch_size = self.batch_size,
                                        sampler = val_sampler,
                                        num_workers = 3)

            return train_dataloader, val_dataloader

        else:
            if os.path.exists(self.preprocessed_dir + "preprocessed_" + train_status + "_" + str(level) + ".pkl"):
                print("load preprocessed dataset " + train_status + " " + str(level))
                with open(self.preprocessed_dir + "preprocessed_" + train_status + "_" + str(level) + ".pkl", "rb") as r_h:
                    test_datasets = pickle.load(r_h)
                
            else:
                _, test_csv_data = self.split_dataset()
                print("create preprocessed dataset " + train_status + " " + str(level))
                test_datasets = self.preprocess_dataset(test_csv_data, level, train_status = train_status)

            test_sampler = RandomSampler(test_datasets)

            test_dataloader = DataLoader(dataset = test_datasets,
                                         batch_size = self.batch_size,
                                         sampler = test_sampler,
                                         num_workers = 3)

            return test_dataloader
        
        
    