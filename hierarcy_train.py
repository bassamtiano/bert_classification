import sys
import argparse
from utils.preprocessor_hierarcy import PreprocessorHierarcy

from utils.tree_helper import TreeHelper

from utils.hierarcy_trainer import HierarcyTrainer

def collect_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_nodes", type=int, default=1)

    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--max_epochs", type=int, default=10)

    parser.add_argument("--embedding_dim", type=int, default=768)

    parser.add_argument("--tree_dir", type=str, default="data/hierarcy/tree/tokopedia.tree")
    parser.add_argument("--dataset_dir", type=str, default="./data/hierarcy/product_tokopedia.csv")
    parser.add_argument("--train_dataset_dir", type=str, default="./data/hierarcy/tokopedia_train.csv")
    parser.add_argument("--test_dataset_dir", type=str, default="./data/hierarcy/tokopedia_test.csv")
    parser.add_argument("--preprocessed_dir", type=str, default="data/hierarcy/preprocessed/")

    parser.add_argument("--n_out", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-5)
    
    return parser.parse_args()

def train_hierachy(args):
    tree_helper = TreeHelper(dir_tree = args.tree_dir)

    tree_level, level_tree, parent2child, level_tree_item, level_tree_ids = tree_helper.load_tree()
    
    layer_len = len(level_tree_ids)

    pre = PreprocessorHierarcy(max_length = args.max_length,
                               batch_size = args.batch_size,
                               dir_dataset = args.dataset_dir,
                               train_dataset_dir = args.train_dataset_dir,
                               test_dataset_dir = args.test_dataset_dir,
                               preprocessed_dir = args.preprocessed_dir,
                               tree_helper = tree_helper,
                               split_data_section = False)

    for i_lyr in range(1, layer_len + 1):

        # pre.preprocessor(level = i_lyr, train_status = "train")

        num_classes = len(level_tree_ids[i_lyr])

        if i_lyr > 1:
            hie_trainer = HierarcyTrainer(level = i_lyr,
                                          preprocess = pre,
                                          embedding_dim = args.embedding_dim,
                                          in_channels = 8,
                                          out_channels = 40,
                                          num_classes = num_classes,
                                          dropout = args.dropout,
                                          lr = args.lr,
                                          device = "cuda",
                                          max_epoch = 20)
            hie_trainer.training()
            break

            
    # pre.load_data(level = 1)

if __name__ == "__main__":
    args = collect_parser()

    train_hierachy(args)