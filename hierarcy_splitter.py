import re

import pandas as pd
from tqdm import tqdm

class HierarchySplitter():

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
        data = pd.read_csv("data/hierarcy/product_tokopedia.csv")

        for i, line in tqdm(enumerate(data.values.tolist())):
            title = self.clean_str(line[0])
            print(title)
            break


if __name__ == '__main__':
    HSplit = HierarchySplitter()
    HSplit.load_data()