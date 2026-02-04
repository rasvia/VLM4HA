import logging
import argparse
import os
import torch
from train_CNN import trainer_CNN

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Training parameters")
parser.add_argument('--data_root', type=str, default='./dataset', help='Directory of dataset')
parser.add_argument('--library', type=str, default='Nangate', help='Name of Standard cell library')
parser.add_argument('--node_technology', type=int, default=45, help='type of node technology')
parser.add_argument('--save_path', type=str, default='./trained_model', help='Directory for saving model weights')
parser.add_argument('--num_epochs', type=int, default=200, help='Training epochs')
parser.add_argument('--num_classes', type=int, default=128, help='Number of classes for label encoding')
parser.add_argument('--num_channels', type=int, default=4, help='Number of channels stacked image')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--max_length', type=int, default=100, help='Max text seq length for tokenization')
parser.add_argument('--max_width', type=int, default=950, help='Max width of image')
parser.add_argument('--test_frac', type=float, default=0.1, help='Fraction of test set') 
parser.add_argument('--es_patience', type=int, default=5, help='Patience for EarlyStopping')
parser.add_argument('--es_delta', type=float, default=0.001, help='minimum delta for early stopping')
parser.add_argument('--vocab_path', type=str, default='./encodings', help='root directory of encoding dictionary')
parser.add_argument('--vocab_size', type=int, default=225, help='number of vocabularies')
parser.add_argument('--test_mode', action="store_true", help='enables testing on trained model')
parser.add_argument('--model_weight', type=str, default=None, help='path to trained ResNet18 weight')



args = parser.parse_args()

config = {'data_dir': os.path.join(*[args.data_root, f'{args.library}{args.node_technology}_cells']),
          'design': args.library,
          'node': args.node_technology,
          'save_path': os.path.join(args.save_path, f'{args.library}{args.node_technology}'),
          'num_epochs': args.num_epochs,
          'num_classes': args.num_classes,
          'num_channels': args.num_channels,
          'lr': args.lr,
          'batch_size': args.batch_size,
          'max_length': args.max_length,
          'max_width': args.max_width,
          'test_frac': args.test_frac,
          'es_patience': args.es_patience,
          'es_delta': args.es_delta,
          'vocab_path': os.path.join(*[args.vocab_path, f'{args.library}_{args.node_technology}nm',
                                       f'{args.library}_{args.node_technology}nm_vocab_dict.pkl']),
          'vocab_size': args.vocab_size,
          'encoding_path': os.path.join(*[args.vocab_path, f'{args.library}_{args.node_technology}nm', 
                                          f'{args.library}_{args.node_technology}nm_cell_encodings.csv']),
          'test_mode': args.test_mode,
          'model_weight': args.model_weight}


def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Initializing for training model......")

    model_trainer = trainer_CNN(config, device)    
    model_trainer.train_model()

def run_testing():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Initializing for testing model......")

    model_trainer = trainer_CNN(config, device)
    model_trainer.test_model()

if __name__ == '__main__':
    if config['test_mode']:
        run_testing()
    else:
        run_training()