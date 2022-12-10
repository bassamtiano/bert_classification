from utils.preprocessor_hierarcy import PreprocessorHierarcy

if __name__ == '__main__':
    
    dm = PreprocessorHierarcy(max_length = 100, 
                              dir_tree = "./data/hierarcy/tree/tokopedia.tree",
                              dir_dataset = "./data/hierarcy/product_tokopedia.csv")

    dm.preprocessor()