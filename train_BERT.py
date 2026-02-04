from model import BERTClassifier
import torch
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import os
from data_utils import MaskedEncodingDataset
import logging
from utils import CustomTokenizer, EarlyStopping
from torch.optim import AdamW
from datetime import datetime
import torch.nn as nn
from sklearn.metrics import accuracy_score
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class trainer_BERT:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.num_epochs = self.config['num_epochs']
        self.lr = self.config['lr']
        self.es_patience = self.config['es_patience']
        self.es_delta = self.config['es_delta']
        self.max_length = self.config['max_length']
        self.batch_size = self.config['batch_size']
        self.save_path = self.config['save_path']
        self.tokenizer = CustomTokenizer(vocab_path=self.config['vocab_path'], vocab_size=self.config['vocab_size'],
                                         max_length=self.max_length, padding=True, return_pt=True)
        self.mlm_weight = 0.5
        self.softmax = torch.nn.Softmax(dim=1)
        if self.config['bert_weight'] is not None:
            self.weight = os.path.join(*[self.save_path, 'BERTClassifier', self.config['bert_weight']])


    def split_dataset(self):
        train_dataset = MaskedEncodingDataset(self.config, 'training')
        valid_dataset = MaskedEncodingDataset(self.config, 'validation')

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)

        return train_loader, valid_loader
    
    def encode_text(self, encodings):
        input_ids, attn_masks = self.tokenizer.batch_encode(encodings)
        input_ids, attn_masks = input_ids.squeeze(1).to(self.device), attn_masks.squeeze(1).to(self.device)

        return input_ids, attn_masks

    def train_model(self):
        logger.info("Starting training model......")

        train_loader, valid_loader = self.split_dataset()

        model = BERTClassifier(self.config).to(self.device)
        best_model_state = self.train_loop(model, train_loader, valid_loader)

        model.load_state_dict(best_model_state)
        logger.info("[Post-training] Best model weights loaded.")

        self.save_model(model)

    def train_epoch(self, model, train_loader, valid_loader, epoch, cls_criterion, mlm_criterion, optimizer, scheduler=None):
        total_loss = 0.0
        scaler = torch.amp.GradScaler()

        for _, batch in enumerate(tqdm(train_loader, desc=f"training at epoch {epoch + 1}/{self.num_epochs}")):
            input_ids, attn_masks = batch['masked_encoding'], batch['attn_mask']
            gt_ids = batch['encoding']

            input_ids, attn_masks = input_ids.squeeze(1).to(self.device), attn_masks.squeeze(1).to(self.device)
            gt_ids =  gt_ids.squeeze(1).to(self.device)
            gt_label = batch['cell_cls'].to(self.device)

            mlm_output, _, logits = model(input_ids, attn_masks)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                mlm_loss = mlm_criterion(mlm_output.view(-1, self.config['vocab_size']), gt_ids.view(-1))
                cls_loss = cls_criterion(logits, gt_label)

                loss = self.mlm_weight * mlm_loss + (1 - self.mlm_weight) * cls_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                if scheduler is not None:
                    scheduler.step()

                total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_valid_loss, val_acc = self.evaluate(model, valid_loader, cls_criterion, mlm_criterion)

        return avg_train_loss, avg_valid_loss, val_acc

    def train_loop(self, model, train_loader, valid_loader):
        early_stopping = EarlyStopping(patience=self.es_patience, min_delta=self.es_delta, verbose=True)
        optimizer = AdamW(model.parameters(), lr=self.lr)
        cls_criterion, mlm_criterion = nn.CrossEntropyLoss(), nn.CrossEntropyLoss(ignore_index=-100)

        for epoch in range(self.num_epochs):
            model.train()
            train_loss, val_loss, acc = self.train_epoch(model, train_loader, valid_loader, epoch, cls_criterion, mlm_criterion, optimizer)

            logger.info(f"[Training] Epoch {epoch + 1} | Train Loss: {train_loss}, Val Loss: {val_loss}, Val acc: {acc}")

            if early_stopping.early_stop(val_loss, model):
                logger.info("[Training] Early Stopping triggered. Stop Training....")
                break

        logger.info(f"[Training] Training Ended.")

        return early_stopping.best_model_state

    def prediction(self, model, batch):
        model.eval()

        input_ids, attn_masks = batch['masked_encoding'], batch['attn_mask']
        gt_ids = batch['encoding']

        input_ids, attn_masks = input_ids.squeeze(1).to(self.device), attn_masks.squeeze(1).to(self.device)
        gt_ids =  gt_ids.squeeze(1).to(self.device)
        gt_label = batch['cell_cls'].to(self.device)

        mlm_output, _, logits = model(input_ids, attn_masks)

        return mlm_output, logits, gt_ids, gt_label

    def evaluate(self, model, valid_loader, cls_criterion, mlm_criterion):
        val_loss = 0.0
        all_preds, all_gt = [], []
        with torch.no_grad():
            for _, batch in enumerate(tqdm(valid_loader, desc=f"Evaluating")):
                mlm_output, logits, gt_ids, gt_label = self.prediction(model, batch)

                mlm_loss = mlm_criterion(mlm_output.view(-1, self.config['vocab_size']), gt_ids.view(-1))
                cls_loss = cls_criterion(logits, gt_label)
                running_loss = self.mlm_weight * mlm_loss + (1 - self.mlm_weight) * cls_loss

                val_loss += running_loss.item()

                preds = torch.argmax(self.softmax(logits), dim=1).cpu()
                gt = gt_label.cpu()

                all_preds += preds
                all_gt += gt

        avg_val_loss = val_loss / len(valid_loader)
        acc = accuracy_score(all_preds, all_gt)
        return avg_val_loss, acc
    
    def test_model(self):
        test_dataset = MaskedEncodingDataset(self.config, 'test')
        model = BERTClassifier(self.config).to(self.device)
        try:
            model.load_state_dict(torch.load(self.weight))
            logger.info("BERT weight loaded.")
        except:
            pass
            logger.info("No model weight found. Loading model with random initialization.")

        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)
        
        features, labels, predictions = self.get_features(model, test_loader)
        acc = accuracy_score(predictions, labels)
        features = np.array(features)
        labels = np.array(labels)
        logger.info(f'Testing acc: {acc}')
        # evaluate_embedding_clusters(features, labels, n_classes=self.config['num_classes'])

    
    def get_features(self, model, test_loader):
        all_features, all_gt, all_pred = [], [], []
        with torch.no_grad():
            for _, batch in enumerate(tqdm(test_loader, desc=f"Predicting")):
                input_ids, attn_masks = batch['text'][0], batch['text'][1]
                input_ids, attn_masks = input_ids.squeeze(1).to(self.device), attn_masks.squeeze(1).to(self.device)
                gt_label = batch['class_label']

                logits, feature = model(input_ids, attn_masks)
                feature, gt = feature.cpu(), gt_label.cpu()

                pred = torch.argmax(self.softmax(logits), dim=1).cpu()

                all_gt += gt
                all_features += feature
                all_pred += pred

        return all_features, all_gt, all_pred
                

    def save_model(self, model):
        timestamp = datetime.now().strftime('%Y%m%d%H%M')
        file_name = f"BERTClassifier_trained_{timestamp}.pt"
        os.makedirs(os.path.join(self.save_path, 'BERTClassifier'), exist_ok=True)
        save_path = os.path.join(*[self.save_path, 'BERTClassifier', file_name])
        logger.info(f"[Post-Training] Saving model to {save_path}...")

        torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), save_path)
        logger.info("[Post-training] Model saved successfully.")