from model import MultiModalVLM
import torch
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import os
from data_utils import CellEncodingDataset
import logging
from utils import EarlyStopping
from torch.optim import AdamW
from datetime import datetime
from torchvision.transforms import ToTensor
import numpy as np
from utils import ArcFaceLoss
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class trainer_VLM:
    def __init__(self, config, device, ablation):
        self.config = config
        self.device = device
        self.design = self.config['design']
        self.node = self.config['node']
        self.num_epochs = self.config['num_epochs']
        self.lr = self.config['lr']
        self.margin = self.config['margin']
        self.scale = self.config['scale']
        self.es_patience = self.config['es_patience']
        self.es_delta = self.config['es_delta']
        self.max_length = self.config['max_length']
        self.batch_size = self.config['batch_size']
        self.save_path = self.config['save_path']
        self.num_classes = self.config['num_classes']
        self.ablation = ablation
        
        self.softmax = torch.nn.Softmax(dim=1)
        if self.config['test_mode']:
            self.weight = os.path.join(*[self.save_path, 'VLM', self.config['model_weight']])
            self.dataset = CellEncodingDataset(self.config, 'test', transform=ToTensor())
            self.gt = CellEncodingDataset(self.config, 'gt', transform=ToTensor())

        self.arcface_head = ArcFaceLoss(in_feature=512, out_feature=self.config['num_classes'], scale=self.scale, margin=self.margin).to(self.device)
        self.arcface_head.train()

    def split_dataset(self):
        train_dataset = CellEncodingDataset(self.config, 'training', transform=ToTensor())
        valid_dataset = CellEncodingDataset(self.config, 'validation', transform=ToTensor())

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)

        return train_loader, valid_loader

    def save_pkl(self, file, filename):
        with open(filename, "wb") as f:
            pickle.dump(file, f)
        f.close()

    def load_pkl(self, file):
        with open(file, 'rb') as f:
            data = pickle.load(f)
        f.close()
        return data

    def train_model(self):
        logger.info("Starting training model......")

        train_loader, valid_loader = self.split_dataset()

        model = MultiModalVLM(self.config).to(self.device)
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        # size_all_mb = (param_size + buffer_size)
        # print(size_all_mb)
        best_model_state = self.train_loop(model, train_loader, valid_loader)

        model.load_state_dict(best_model_state)
        logger.info("[Post-training] Best model weights loaded.")

        self.save_model(model)


    def train_epoch(self, model, train_loader, valid_loader, epoch, criterion, optimizer, scheduler=None):
        total_loss = 0.0

        for _, batch in enumerate(tqdm(train_loader, desc=f"training at epoch {epoch + 1}/{self.num_epochs}")):
            image = batch['image'].float().to(self.device)
            input_ids, attn_mask = batch['text'][0].squeeze(1).to(self.device), batch['text'][1].squeeze(1).to(self.device)
            labels = batch['cell_cls'].to(self.device)

            embedding = model(image, input_ids, attn_mask)
            logits = self.arcface_head(embedding, labels)
            optimizer.zero_grad()

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.arcface_head.parameters(), max_norm=1.0)

            if scheduler is not None:
                scheduler.step()
            
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        # avg_valid_loss, auc, eer = self.evaluate(model, valid_loader, criterion)
        # return avg_train_loss, avg_valid_loss, auc, eer
        avg_valid_loss = self.evaluate(model, valid_loader, criterion)
        return avg_train_loss, avg_valid_loss

    def train_loop(self, model, train_loader, valid_loader):
        early_stopping = EarlyStopping(patience=self.es_patience, min_delta=self.es_delta, verbose=True)
        criterion = nn.CrossEntropyLoss()

        optimizer = AdamW([{'params': model.parameters()},
                        {'params': self.arcface_head.parameters()}], lr=self.lr, weight_decay=1e-4)

        for epoch in range(self.num_epochs):
            model.train()
            # train_loss, val_loss, auc, eer = self.train_epoch(model, train_loader, valid_loader, epoch, criterion, optimizer)
            # logger.info(f"[Training] Epoch {epoch + 1} | Train Loss: {train_loss}, Val Loss: {val_loss}, AUC-ROC: {auc}, EER: {eer}")

            train_loss, val_loss = self.train_epoch(model, train_loader, valid_loader, epoch, criterion, optimizer)
            logger.info(f"[Training] Epoch {epoch + 1} | Train Loss: {train_loss}, Val Loss: {val_loss}")


            if early_stopping.early_stop(val_loss, model):
                logger.info("[Training] Early Stopping triggered. Stop Training....")
                break

        logger.info(f"[Training] Training Ended.")

        return early_stopping.best_model_state

    def prediction(self, model, batch):
        model.eval()

        image = batch['image'].float().to(self.device)
        input_ids, attn_mask = batch['text'][0].squeeze(1).to(self.device), batch['text'][1].squeeze(1).to(self.device)
        labels = batch['cell_cls'].to(self.device)

        embedding = model(image, input_ids, attn_mask)
        logits = self.arcface_head(embedding, labels)

        return logits, labels

    def evaluate(self, model, valid_loader, criterion):
        val_loss = 0.0

        embeddings, labels = [], []

        with torch.no_grad():
            for _, batch in enumerate(tqdm(valid_loader, desc=f"Evaluating")):
                embedding, label = self.prediction(model, batch)

                embeddings.extend(embedding.cpu().tolist())
                labels.extend(label.cpu().tolist())
                running_loss = criterion(embedding, label)

                val_loss += running_loss.item()

        avg_val_loss = val_loss / len(valid_loader)
        # auc, eer = self.compute_metrics(embeddings, labels)
        # return avg_val_loss, auc, eer
        return avg_val_loss
    
    def compute_metrics(self, embeddings, labels):
        embeddings = np.array(embeddings)
        labels = np.array(labels)
        
        num_samples = len(embeddings)
        if num_samples < 2:
            return 0.5, 0.5 
        
        similarity_matrix = np.dot(embeddings, embeddings.T)
        label_matrix = labels.reshape(-1, 1) == labels.reshape(1, -1)
        
        triu_indices = np.triu_indices(num_samples, k=1)
        
        all_scores = similarity_matrix[triu_indices]
        all_matches = label_matrix[triu_indices]
        
        positive_scores = all_scores[all_matches]
        negative_scores = all_scores[~all_matches]
        
        num_pos = len(positive_scores)
        num_neg = len(negative_scores)

        if num_pos == 0 or num_neg == 0:
            return 0.5, 0.5 

        num_to_sample = min(num_pos, num_neg)
        
        positive_samples = np.random.choice(positive_scores, size=num_to_sample, replace=False)
        negative_samples = np.random.choice(negative_scores, size=num_to_sample, replace=False)
        
        y_true = np.hstack([np.ones(num_to_sample), np.zeros(num_to_sample)])
        y_scores = np.hstack([positive_samples, negative_samples])

        auc = roc_auc_score(y_true, y_scores)
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)      

        return auc, eer
    
    def test_model(self):
        model = MultiModalVLM(self.config, ablation=self.ablation).to(self.device)

        try:
            model.load_state_dict(torch.load(self.weight))
            logger.info("[Testing] VLM weight loaded.")
        except:
            pass
            logger.info("[Testing] No model weight found. Loading model with random weight.")

        model.eval()

        test_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)
        
        results = self.get_features(model, test_loader)

        with open(f'./results/{self.design}{self.node}/VLM/VLM_results_{self.ablation}.pkl', "wb") as f:
            pickle.dump(results, f)
        f.close()
        logger.info('[Testing] results saved.')

    def get_embedding(self, model, batch):
        image = batch['image'].float().to(self.device)
        input_ids, attn_mask = batch['text'][0].squeeze(1).to(self.device), batch['text'][1].squeeze(1).to(self.device)

        embedding = model(image, input_ids, attn_mask)

        return embedding

    def build_gallery(self, model):
        if os.path.exists(f'./results/{self.design}{self.node}/GRU_gallery_{self.ablation}.pkl'):
            logger.info('[Testing] Existing gallery found, loading...')
            gallery_file = self.load_pkl(file=f'./results/{self.design}{self.node}/GRU_gallery_{self.ablation}.pkl')
            gallery = gallery_file['gallery']
            gallery_class = gallery_file['gallery_class']
            logger.info('[Testing] Gallery loaded. ')
        else:
            logger.info('[Testing] No existing gallery found. ')
            gallery_loader, _ = self.split_dataset()
            gallery, gallery_class = [], []

            for _, batch in enumerate(tqdm(gallery_loader, desc='[Testing] Building gallery')):
                embedding = self.get_embedding(model, batch)
                gallery.extend(embedding.cpu().tolist())
                gallery_class.extend(batch['cell_cls'].cpu().tolist())

            gallery = np.array(gallery)
            gallery_class = np.array(gallery_class)

            gallery_file = {'gallery': gallery, 'gallery_class': gallery_class}

            self.save_pkl(gallery_file, filename=f'./results/{self.design}{self.node}/concat_gallery_{self.ablation}.pkl')
            logger.info('[Testing] Gallery file saved.')

        return gallery, gallery_class


    def get_features(self, model, test_loader):
        gallery, gallery_class = self.build_gallery(model)

        test_embeddings, test_labels, class_labels = [], [], []
        with torch.no_grad():
            for _, batch in enumerate(tqdm(test_loader, desc=f'Testing')):
                labels, anomaly = batch['cell_cls'].to(self.device), batch['label'].to(self.device)
                embedding = self.get_embedding(model, batch)
                test_embeddings.extend(embedding.cpu().tolist())
                test_labels.extend(anomaly.cpu().tolist())
                class_labels.extend(labels.tolist())

        test_embeddings = np.array(test_embeddings)
        test_labels = np.array(test_labels)
        class_labels = np.array(class_labels)
        all_scores_cls = []
        all_scores = []
        logger.info('[Testing] Calculating similarity scores...')
        for i in tqdm(range(0, test_embeddings.shape[0]), desc=f'Computing'):
            cls_label = class_labels[i]

            scores_per_sample = []
            for cls in range(0, self.num_classes):
                sub_gallery = gallery[gallery_class == cls]
                score_per_class = np.max(np.clip(np.dot(test_embeddings[i], sub_gallery.T), 0, 1))
                # score_per_class = np.max(1 / (1+np.linalg.norm(test_embeddings[i] - sub_gallery, axis=1)))
                # score_per_class = np.max(np.exp(-np.linalg.norm(test_embeddings[i] - sub_gallery, axis=1)))
                scores_per_sample.append(score_per_class)

            all_scores.append(scores_per_sample)
            all_scores_cls.append(scores_per_sample[cls_label])
        
        logger.info('[Testing] Similarity scores computed.')

        results = {'similarity_score': all_scores_cls, 
                   'similarity_score_all_class': all_scores,
                   'anomaly_label': test_labels.tolist(), 
                   'cell_label': class_labels.tolist(),
                   }
        return results

    def save_model(self, model):
        timestamp = datetime.now().strftime('%Y%m%d%H%M')
        file_name = f"VLM_concat_trained_{timestamp}.pt"
        os.makedirs(os.path.join(self.save_path, 'VLM'), exist_ok=True)
        save_path = os.path.join(*[self.save_path, 'VLM', file_name])
        logger.info(f"[Post-Training] Saving model to {save_path}...")

        torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), save_path)
        logger.info("[Post-training] Model saved successfully.")
