import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
import torch
import torchvision.models as models
import os
from torch.nn.utils.rnn import pack_padded_sequence

class ResNet18(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = self.config['num_classes']
        self.num_channels = self.config['num_channels']
        self.resnet = models.resnet18(weights=None)
        original_conv1 = self.resnet.conv1

        self.resnet.conv1 = nn.Conv2d(self.num_channels,
                                      original_conv1.out_channels,
                                      kernel_size=original_conv1.kernel_size,
                                      stride=original_conv1.stride,
                                      padding=original_conv1.padding,
                                      bias=original_conv1.bias)
        
        self.hidden_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.num_classes)

        self.feature_extractor = torch.nn.Sequential(*list(self.resnet.children())[:-1])

    def forward(self, x):
        output = self.resnet(x)
        feature = self.feature_extractor(x)

        return output, feature
    

class BERTClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = self.config['num_classes']

        bert_config = AutoConfig.from_pretrained('bert-base-uncased')
        self.bert = AutoModel.from_config(bert_config)

        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_classes)
        self.dropout = nn.Dropout(0.1)

        self.mlm = nn.Linear(self.bert.config.hidden_size, self.config['vocab_size'])

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        mlm_output = self.mlm(output.last_hidden_state)
        cls_output = output.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)

        return mlm_output, cls_output, logits


class GRUClassifier(nn.Module):
    def __init__(self, config, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.config = config
        self.vocab_size = self.config['vocab_size']
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_classes = self.config['num_classes']
        self.pad_idx = 0

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.pad_idx)
        self.gru = nn.GRU(self.embed_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_dim * 2, self.n_classes)

    def forward(self, input_ids, attention_mask):
        lengths = attention_mask.sum(dim=1).cpu()

        x = self.embedding(input_ids)
        x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        _, hidden = self.gru(x_packed)
        combined = torch.cat((hidden[-2,:,:], hidden[-1, :, :]), dim=1)

        return self.fc(combined), combined, lengths

class MultiModalVLM(nn.Module):
    def __init__(self, config, ablation='full'):
        super().__init__()
        self.config = config
        self.save_path = self.config['save_path']
        self.shared_embed_dim = 512

        self.image_encoder = ResNet18(self.config)

        if 'BERT' in self.config['bert_weight']:
            self.text_encoder = BERTClassifier(self.config)
            self.text_projection = nn.Linear(self.text_encoder.bert.config.hidden_size, self.shared_embed_dim)
        else:
            self.text_encoder = GRUClassifier(self.config)
            self.text_projection = nn.Linear(self.text_encoder.hidden_dim * 2, self.shared_embed_dim)

        if self.config['bert_weight'] is not None:
            if 'BERT' in self.config['bert_weight']:
                self.bert_weight = os.path.join(*[self.save_path, 'BERTClassifier', self.config['bert_weight']])
            else:
                self.bert_weight = os.path.join(*[self.save_path, 'GRUClassifier', self.config['bert_weight']])
            self.text_encoder.load_state_dict(torch.load(self.bert_weight))
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            
        if self.config['cnn_weight'] is not None:
            self.cnn_weight = os.path.join(*[self.save_path, 'ResNet18', self.config['cnn_weight']])
            self.image_encoder.load_state_dict(torch.load(self.cnn_weight), strict=False)
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        self.image_projection = nn.Linear(self.image_encoder.hidden_dim, self.shared_embed_dim)
        
        self.gate_network = nn.Sequential(
            nn.Linear(self.shared_embed_dim * 2, self.shared_embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.shared_embed_dim * 4, self.shared_embed_dim),
            nn.Sigmoid()
        )

        self.norm_t = nn.LayerNorm(self.shared_embed_dim)
        self.norm_i = nn.LayerNorm(self.shared_embed_dim)

        self.ablation = ablation

    def forward(self, image, input_ids, attn_masks):
        _, image_embed = self.image_encoder(image)
        _, text_embed, _ = self.text_encoder(input_ids, attn_masks)

        image_embed = image_embed.squeeze(3).squeeze(2)

        if self.ablation == 'image_only':
            text_embed = torch.zeros_like(text_embed)
        elif self.ablation == 'encoding_only':
            image_embed = torch.zeros_like(image_embed)
            
        proj_i = self.norm_i(self.image_projection(image_embed))
        proj_t = self.norm_t(self.text_projection(text_embed))
        concat_embed = torch.cat((proj_i, proj_t), dim=1)

        z = self.gate_network(concat_embed)

        fused_embedding = z * proj_i + (1 - z) * proj_t
        fused_embedding = F.normalize(fused_embedding, p=2, dim=1, eps=1e-7)

        return fused_embedding


