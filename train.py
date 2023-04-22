from networks import *
from encoders import *
from dataset import *
from sklearn.model_selection import train_test_split

from dataclasses import dataclass
from typing import Tuple, Literal
import tyro

import torch
import torch.nn as nn
import os
import os.path as path
import pandas as pd
import torch.optim as optim
from tqdm import tqdm, trange
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassAccuracy, MulticlassRecall, MulticlassF1Score

import matplotlib.pyplot as plt
import seaborn as sns

@dataclass(frozen=True)
class cfg:
    n_epochs: int = 500
    network: Literal['CNN', 'LSTM'] = 'LSTM'
    bidirectional: bool = False
    test_size: float = 0.2
    d_input: int = 1
    d_hidden: int = 128
    n_layers: int = 1
    n_classes: int  = 5
    n_batches: int = 1024
    lr: float = 0.001
    n_freqs: int = 8
    device: str = 'cuda:0'
    log_dir: str = './logs'
    exp_tag: str = ''

if __name__ == '__main__':
    cfg = tyro.cli(cfg)

    train_df = pd.read_csv('./data/train.csv')
    test_df = pd.read_csv('./data/test.csv')
    target_col = '187'
    features_col = [x for x in train_df.columns if x != '187']
    X = np.asarray(train_df[features_col])
    y = np.asarray(train_df[target_col])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=cfg.test_size, random_state=42)
    print(X.shape, y.shape, X_train.shape, X_val.shape, y_train.shape, y_val.shape)

    exp_name = f'network={cfg.network}_d-hidden={cfg.d_hidden}_n-layers={cfg.n_layers}_bidirectional={cfg.bidirectional}_n-freqs={cfg.n_freqs}'
    if cfg.exp_tag != '':
        exp_name = cfg.exp_tag + '-' + exp_name
    log_path = path.join(cfg.log_dir, exp_name)
    writer = SummaryWriter(log_dir=log_path)

    # instantiating the dataset and dataloader for both training and validation
    heart_training = HeartBeatDataset(X_train, y_train)
    heart_validating = HeartBeatDataset(X_val, y_val)

    heart_trainloader = DataLoader(heart_training, batch_size=cfg.n_batches, shuffle=True)
    heart_valloader = DataLoader(heart_validating, batch_size=cfg.n_batches, shuffle=True)

    # instantiating the network, loss function, and optimizer
    encoder = PositionalEncoder(cfg.d_input, cfg.n_freqs).to(cfg.device)

    if cfg.network == 'LSTM':
        network = LSTM(encoder.d_output, cfg.d_hidden, cfg.n_classes, cfg.n_layers, cfg.bidirectional).to(cfg.device)
    elif cfg.network == 'CNN':
        network = CNN(encoder.d_output, cfg.d_hidden, cfg.n_classes, cfg.n_layers).to(cfg.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=cfg.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    # to store the loss
    loss_list = list()

    # training the network
    for epoch in trange(cfg.n_epochs):
        network.train()
        running_loss = 0.0
        
        for i, data in enumerate(heart_trainloader):
            inputs, labels = data
            inputs = inputs.to(cfg.device)
            labels = labels.long().to(cfg.device)
            optimizer.zero_grad()
            inputs = encoder(inputs.float())
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        network.eval()
            
        val_loss = 0.0
        acc = 0.0
        recall = 0.0
        f1 = 0.0
        confusion_mats = []
        
        acc_fn = MulticlassAccuracy(num_classes=cfg.n_classes).to(cfg.device)
        recall_fn = MulticlassRecall(num_classes=cfg.n_classes).to(cfg.device)
        f1_fn = MulticlassF1Score(num_classes=cfg.n_classes).to(cfg.device)
        confusion_mat_fn = MulticlassConfusionMatrix(num_classes=cfg.n_classes).to(cfg.device)
        # acc = 0.0
        with torch.no_grad():
            for i, data in enumerate(heart_valloader):
                inputs, labels = data
                inputs = inputs.to(cfg.device)
                labels = labels.long().to(cfg.device)
                inputs = encoder(inputs.float())
                outputs = network(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                acc += acc_fn(outputs, labels)
                recall += recall_fn(outputs, labels)
                f1 += f1_fn(outputs, labels)
                confusion_mats.append(confusion_mat_fn(outputs, labels))
                
                
        train_loss = running_loss/len(heart_training)
        val_loss = val_loss/len(heart_validating)
        acc = acc / (i+1)
        recall = recall / (i+1)
        f1 = f1 / (i+1)
        confusion_mat = torch.stack(confusion_mats, dim=0).float().mean(dim=0).cpu().detach().numpy()
        classes = ['Normal', 'Supraventricular', 'Ventricular', 'Ventricular + normal', 'Unclassifiable']
        df_cm = pd.DataFrame(confusion_mat / np.sum(confusion_mat, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
        plt.figure(figsize=(12, 7), dpi=120) 
        
        writer.add_figure("Confusion matrix", sns.heatmap(df_cm, annot=True, cmap='coolwarm_r').get_figure(), epoch)
        writer.add_scalar(f'Loss/Train loss', train_loss, global_step=epoch)
        writer.add_scalar(f'Loss/Val loss', val_loss, global_step=epoch)
        writer.add_scalar(f'Metrics/Accuracy', acc, global_step=epoch)
        writer.add_scalar(f'Metrics/Recall', recall, global_step=epoch)
        writer.add_scalar(f'Metrics/F1', f1, global_step=epoch)
        
        loss_list.append((train_loss, val_loss))
        
        str_epoch = f'Epoch [{epoch+1}/{cfg.n_epochs}]'
        str_train_loss = f'Training Loss: {train_loss:.8f}'
        str_val_loss = f'Validation Loss: {val_loss:.8f}'
        str_acc_value = f'Accuracy: {acc:.4f}'
        str_rec_value = f'Recall: {recall:.4f}'
        str_f1_value = f'F1: {f1:.4f}'
        
        tqdm.write(f'{str_epoch}, {str_train_loss}, {str_val_loss}, {str_acc_value}, {str_rec_value}, {str_f1_value}')

