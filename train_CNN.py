from model import ResNet18
import torch
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import os
from data_utils import CellImageDataset
import logging
from utils import EarlyStopping
from torch.optim import AdamW
from datetime import datetime
import torch.nn as nn
from torchvision.transforms import ToTensor
from sklearn.metrics import accuracy_score
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from cluster_eval import evaluate_embedding_clusters

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class trainer_CNN:
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
        
        self.softmax = torch.nn.Softmax(dim=1)
        if self.config['test_mode']:
            self.weight = os.path.join(*[self.save_path, 'ResNet18', self.config['model_weight']])
            self.dataset = CellImageDataset(self.config, 'test', transform=ToTensor())

    def split_dataset(self):
        train_dataset = CellImageDataset(self.config, 'training', transform=ToTensor())
        valid_dataset = CellImageDataset(self.config, 'validation', transform=ToTensor())

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)
        return train_loader, valid_loader


    def train_model(self):
        logger.info("Starting training model......")

        train_loader, valid_loader = self.split_dataset()

        model = ResNet18(self.config).to(self.device)
        best_model_state = self.train_loop(model, train_loader, valid_loader)

        model.load_state_dict(best_model_state)
        logger.info("[Post-training] Best model weights loaded.")

        self.save_model(model)

    def train_epoch(self, model, train_loader, valid_loader, epoch, criterion, optimizer, scheduler=None):
        total_loss = 0.0
        scaler = torch.amp.GradScaler()

        for _, batch in enumerate(tqdm(train_loader, desc=f"training at epoch {epoch + 1}/{self.num_epochs}")):
            image = batch['image'].float().to(self.device)
            gt_label = batch['cell_class'].to(self.device)

            output, _ = model(image)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                loss = criterion(output, gt_label)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                if scheduler is not None:
                    scheduler.step()

                total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_valid_loss, val_acc = self.evaluate(model, valid_loader, criterion)

        return avg_train_loss, avg_valid_loss, val_acc

    def train_loop(self, model, train_loader, valid_loader):
        early_stopping = EarlyStopping(patience=self.es_patience, min_delta=self.es_delta, verbose=True)
        optimizer = AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=5)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.num_epochs):
            model.train()
            train_loss, val_loss, acc = self.train_epoch(model, train_loader, valid_loader, epoch, criterion, optimizer, scheduler)

            logger.info(f"[Training] Epoch {epoch + 1} | Train Loss: {train_loss}, Val Loss: {val_loss}, Val acc: {acc}")

            if early_stopping.early_stop(val_loss, model):
                logger.info("[Training] Early Stopping triggered. Stop Training....")
                break

        logger.info(f"[Training] Training Ended.")

        return early_stopping.best_model_state

    def prediction(self, model, batch):
        model.eval()

        image = batch['image'].float().to(self.device)
        gt_label = batch['cell_class'].to(self.device)

        output, _ = model(image)

        return output, gt_label

    def evaluate(self, model, valid_loader, criterion):
        val_loss = 0.0
        all_preds, all_gt = [], []
        with torch.no_grad():
            for _, batch in enumerate(tqdm(valid_loader, desc=f"Evaluating")):
                predict, gt_label = self.prediction(model, batch)
                running_loss = criterion(predict, gt_label)

                val_loss += running_loss.item()

                preds = torch.argmax(self.softmax(predict), dim=1).cpu()
                gt = gt_label.cpu()

                all_preds += preds
                all_gt += gt

        avg_val_loss = val_loss / len(valid_loader)
        acc = accuracy_score(all_preds, all_gt)
        return avg_val_loss, acc
    
    def test_model(self):
        model = ResNet18(self.config).to(self.device)
        try:
            model.load_state_dict(torch.load(self.weight))
            logger.info("ResNet weight loaded.")
        except:
            pass
            logger.info("No model weight found. Loading model with random initialization.")

        test_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)
        
        features, labels, predictions = self.get_features(model, test_loader)
        acc = accuracy_score(predictions, labels)
        features = np.array(features)
        labels = np.array(labels)

        print(features.shape, labels.shape)
        logger.info(f'Testing acc: {acc}')
        evaluate_embedding_clusters(features, labels, n_classes=self.config['num_classes'])

    
    def get_features(self, model, test_loader):
        all_features, all_gt, all_pred = [], [], []
        with torch.no_grad():
            for _, batch in enumerate(tqdm(test_loader, desc=f"Predicting")):
                image = batch['image'].float().to(self.device)
                gt_label = batch['cell_class']

                logits, feature = model(image)
                feature, gt = feature.squeeze(3).squeeze(2).cpu(), gt_label.cpu()

                pred = torch.argmax(self.softmax(logits), dim=1).cpu()

                all_gt += gt
                all_features += feature
                all_pred += pred

        return all_features, all_gt, all_pred

    def save_model(self, model):
        timestamp = datetime.now().strftime('%Y%m%d%H%M')
        file_name = f"ResNet18_trained_{timestamp}.pt"
        os.makedirs(os.path.join(self.save_path, 'ResNet18'), exist_ok=True)
        save_path = os.path.join(*[self.save_path, 'ResNet18', file_name])
        logger.info(f"[Post-Training] Saving model to {save_path}...")

        torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), save_path)
        logger.info("[Post-training] Model saved successfully.")