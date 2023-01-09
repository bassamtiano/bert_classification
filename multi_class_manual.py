import argparse

from utils.preprocessor_class import PreprocessorClass
from utils.trainer import Trainer

def collect_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_nodes", type=int, default=1)

    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--max_epochs", type=int, default=10)

    parser.add_argument("--train_data_dir", type=str, default="data/multiclass/training.res")
    parser.add_argument("--test_data_dir", type=str, default="data/multiclass/testing.res")
    parser.add_argument("--preprocessed_dir", type=str, default="data/multiclass/preprocessed")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/bert-multi-class")
    parser.add_argument("--log_name", type=str, default="bert-multi-class")

    parser.add_argument("--n_out", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-5)
    
    return parser.parse_args()

if __name__ == '__main__':

    args = collect_parser()

    pre = PreprocessorClass(max_length = args.max_length, 
                            preprocessed_dir = args.preprocessed_dir,
                            train_data_dir = args.train_data_dir,
                            test_data_dir = args.test_data_dir,
                            batch_size = args.batch_size)

    trainer = Trainer(preprocessor=pre, device = "cuda:0", n_out = args.n_out, dropout = args.dropout, lr = args.lr)
    trainer.training()

    