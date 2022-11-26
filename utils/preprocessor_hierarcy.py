import re
import pandas as pd

from transformers import BertTokenizer

class PreprocessorHierarcy():
    def __init__(self, max_length = 100, batch_size = 40):
        super(PreprocessorHierarcy, self).__init__()
        self.test = "a"

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

    def encode_text(self, text):
        tkn = self.tokenizers(text = text,
                             max_length = self.max_length,
                             padding = 'max_length',
                             truncation = True)
        return tkn['input_ids'], tkn['token_type_ids'], tkn['attention_mask']

    def load_tree(self):
        tree_info = {}
        with open("data/hierarcy/tree/tokopedia.tree", "r") as f:
            for line in f:
                line = line[:-1]
                category = line.split(" > ")[-1]
                level = len(line.split(" > "))
                if category not in tree_info:
                    tree_info[category] = level
        return tree_info

    def load_data(self, ):
        tree_info = self.load_tree()
        max_sen_len = 0
        data_list = []

        data = pd.read_csv("data/hierarcy/data_tokopedia.csv")
        for line in data.values.tolist():
            temp_dict = {}

            input_ids, token_type_ids, attention_mask = self.encode_text(line[0])

            kategori = line[3]
            hierarcy = [set([tree_info[cat] for cat in kategori.split(" > ")])]
            category = [cat for cat in kategori.split(" > ")]
            # data_list.append()  
            print(category)
            
        
        


if __name__ == '__main__':
    pre = PreprocessorHierarcy()
    pre.load_data()