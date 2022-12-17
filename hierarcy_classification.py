from utils.preprocessor_hierarcy import PreprocessorHierarcy

def collect_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_nodes", type=int, default=1)

    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--max_epochs", type=int, default=10)

    parser.add_argument("--tree_dir", type=str, default="data/hierarcy/tree/tokopedia.tree")
    parser.add_argument("--train_data_dir", type=str, default="data/hierarcy/train.res")
    parser.add_argument("--test_data_dir", type=str, default="data/hierarcy/testing.res")
    parser.add_argument("--preprocessed_dir", type=str, default="data/multiclass/preprocessed")

    parser.add_argument("--n_out", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-5)
    
    return parser.parse_args()

if __name__ == '__main__':

    args = collect_parser()
    
    dm = PreprocessorHierarcy(max_length = 100, 
                              dir_tree = args.tree_dir,
                              dir_dataset = "./data/hierarcy/product_tokopedia.csv")

    dm.preprocessor()

