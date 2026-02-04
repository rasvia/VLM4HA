import logging
import argparse
import os
import torch
from train_VLM import trainer_VLM

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Training parameters")
parser.add_argument('--data_root', type=str, default='./dataset', help='Directory of dataset')
parser.add_argument('--library', type=str, default='SAED', help='Name of Standard cell library')
parser.add_argument('--node_technology', type=int, default=32, help='type of node technology')
parser.add_argument('--save_path', type=str, default='./trained_model', help='Directory for saving model weights')
parser.add_argument('--num_epochs', type=int, default=200, help='Training epochs')
parser.add_argument('--num_classes', type=int, default=286, help='Number of classes for label encoding')
parser.add_argument('--num_channels', type=int, default=4, help='Number of channels stacked image')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--margin', type=float, default=0.5, help='margin for arcface loss')
parser.add_argument('--scale', type=float, default=30.0, help='scale for arcface loss')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--max_length', type=int, default=105, help='Max text seq length for tokenization')
parser.add_argument('--max_width', type=int, default=1950, help='Max width of image')
parser.add_argument('--es_patience', type=int, default=5, help='Patience for EarlyStopping')
parser.add_argument('--es_delta', type=float, default=0.0, help='minimum delta for early stopping')
parser.add_argument('--vocab_path', type=str, default='./encodings', help='root directory of encoding dictionary')
parser.add_argument('--vocab_size', type=int, default=245, help='number of vocabularies')
parser.add_argument('--test_mode', action="store_true", help='enables testing on trained model')
parser.add_argument('--BERT_weight', type=str, default=None, help='path to trained BERT weight')
parser.add_argument('--CNN_weight', type=str, default=None, help='path to trained ResNet18 weight')
parser.add_argument('--model_weight', type=str, default=None, help='path to trained VLM weight')

args = parser.parse_args()

config = {'data_dir': os.path.join(*[args.data_root, f'{args.library}{args.node_technology}_cells']),
          'design': args.library,
          'node': args.node_technology,
          'save_path': os.path.join(args.save_path, f'{args.library}{args.node_technology}'),
          'num_epochs': args.num_epochs,
          'num_classes': args.num_classes,
          'num_channels': args.num_channels,
          'lr': args.lr,
          'margin': args.margin,
          'scale': args.scale,
          'batch_size': args.batch_size,
          'max_length': args.max_length,
          'max_width': args.max_width,
          'es_patience': args.es_patience,
          'es_delta': args.es_delta,
          'vocab_path': os.path.join(*[args.vocab_path, f'{args.library}_{args.node_technology}nm',
                                       f'{args.library}_{args.node_technology}nm_vocab_dict.pkl']),
          'vocab_size': args.vocab_size,
          'encoding_path': os.path.join(*[args.vocab_path, f'{args.library}_{args.node_technology}nm', 
                                          f'{args.library}_{args.node_technology}nm_cell_encodings.csv']),
          'test_mode': args.test_mode,
          'bert_weight': args.BERT_weight,
          'cnn_weight': args.CNN_weight,
          'model_weight': args.model_weight,
          }

def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Initializing for training model...")

    model_trainer = trainer_VLM(config, device, ablation='full')
    
    model_trainer.train_model()

def run_testing():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Initializing for testing model...")

    model_trainer = trainer_VLM(config, device, ablation='image_only')
    model_trainer.test_model()

if __name__ == '__main__':
    if config['test_mode']:
        run_testing()
    else:
        run_training()
