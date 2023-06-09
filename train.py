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
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassAccuracy, MulticlassRecall, MulticlassF1Score, ROC
from imblearn.over_sampling import ADASYN

import matplotlib.pyplot as plt
import seaborn as sns

import pickle as pkl

@dataclass(frozen=True)
class cfg:
    n_epochs: int = 1000
    mode: Literal["train", "test"] = "train"
    network: Literal['CNN', 'LSTM', 'CNN_LSTM'] = 'CNN'
    bidirectional: bool = False
    test_size: float = 0.2
    d_input: int = 1
    d_hidden: int = 128
    n_layers: int = 1
    n_classes: int  = 5
    n_batches: int = 1024
    weighted_loss: bool = True
    lr: float = 0.001
    n_freqs: int = 4
    i_val: int = 10
    i_save: int = 100
    i_pred: int = 10
    device: str = 'cuda:0'
    log_dir: str = './logs'
    exp_tag: str = ''
    resample: str = 'minority'
    
def set_all_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    set_all_seeds()
    cfg = tyro.cli(cfg)

    train_df = pd.read_csv('./data/train.csv')
    test_df = pd.read_csv('./data/test.csv')
    target_col = '187'
    features_col = [x for x in train_df.columns if x != '187']
    X = np.asarray(train_df[features_col])
    y = np.asarray(train_df[target_col])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=cfg.test_size, random_state=42)
    if cfg.mode == 'test':
        X_train = X
        y_train = y

    X_bal, y_bal = ADASYN(sampling_strategy=cfg.resample,random_state=42).fit_resample(X_train, y_train)

    print(X.shape, y.shape, X_train.shape, X_val.shape, y_train.shape, y_val.shape, X_bal.shape, y_bal.shape)

    exp_name = f'{cfg.network}_d-hidden={cfg.d_hidden}_n-layers={cfg.n_layers}_n-freqs={cfg.n_freqs}_sample_strg={cfg.resample}'

    if not cfg.weighted_loss:
        exp_name += f'_no-weighted-loss'
    if cfg.bidirectional:
        exp_name += f'_bidirectional'
    if cfg.exp_tag != '':
        exp_name = cfg.exp_tag + '-' + exp_name
    log_path = path.join(cfg.log_dir, exp_name)
    writer = SummaryWriter(log_dir=log_path)

    # instantiating the dataset and dataloader for both training and validation
    # heart_training = HeartBeatDataset(X_train, y_train)
    heart_training = HeartBeatDataset(X_bal, y_bal)
    heart_validating = HeartBeatDataset(X_val, y_val)

    heart_trainloader = DataLoader(heart_training, batch_size=cfg.n_batches, shuffle=True)
    heart_valloader = DataLoader(heart_validating, batch_size=cfg.n_batches, shuffle=False)

    # instantiating the network, loss function, and optimizer
    encoder = PositionalEncoder(cfg.d_input, cfg.n_freqs).to(cfg.device)

    if cfg.network == 'LSTM':
        network = LSTM(encoder.d_output, cfg.d_hidden, cfg.n_classes, cfg.n_layers, cfg.bidirectional).to(cfg.device)
    elif cfg.network == 'CNN':
        network = CNN(encoder.d_output, cfg.d_hidden, cfg.n_classes, cfg.n_layers).to(cfg.device)
    elif cfg.network == 'CNN_LSTM':
        network = CNN_LSTM(encoder.d_output, cfg.d_hidden, cfg.n_classes, cfg.n_layers, cfg.bidirectional).to(cfg.device)
        
    if cfg.weighted_loss:
        n_per_class = pd.Series(y_train).value_counts().sort_index()
        weight = (n_per_class.max() / n_per_class).to_list()
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weight)).to(cfg.device)
    else:
        criterion = nn.CrossEntropyLoss().to(cfg.device)
        
    optimizer = optim.Adam(network.parameters(), lr=cfg.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    # training the network
    for i_epoch in trange(1, cfg.n_epochs+1):
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
        tqdm.write(f'[Epoch {i_epoch}]: Loss = {running_loss/(i+1)}')
        
        if i_epoch % cfg.i_val == 0:
            val_loss = 0.0
            acc = 0.0
            recall = 0.0
            confusion_mats = []
            label_list = []
            pred_list = []
            
            acc_fn = MulticlassAccuracy(num_classes=cfg.n_classes).to(cfg.device)
            recall_fn = MulticlassRecall(num_classes=cfg.n_classes).to(cfg.device)
            confusion_mat_fn = MulticlassConfusionMatrix(num_classes=cfg.n_classes).to(cfg.device)
            roc_curve = ROC(task="multiclass", num_classes=cfg.n_classes).to(cfg.device)

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
                    confusion_mats.append(confusion_mat_fn(outputs, labels))
                    label_list.extend(labels)
                    pred_list.extend(outputs)
                    
            train_loss = running_loss/len(heart_training)
            val_loss = val_loss/len(heart_validating)
            # acc = acc / (i+1)
            # recall = recall / (i+1)
            # f1 = 2*acc*recall / (acc + recall)
            confusion_mat = torch.stack(confusion_mats, dim=0).float().sum(dim=0).cpu().detach().numpy()
            classes = ['Normal', 'Supraventricular', 'Ventricular', 'Ventricular + normal', 'Unclassifiable']
            df_cm = pd.DataFrame(confusion_mat / np.sum(confusion_mat, axis=1)[:, None], index=[i for i in classes],
                            columns=[i for i in classes])
            plt.figure(figsize=(12, 7), dpi=120) 
            recall = (np.diag(confusion_mat) / np.sum(confusion_mat, axis = 1)).mean()
            acc = (np.diag(confusion_mat) / np.sum(confusion_mat, axis = 0)).mean()
            f1 = 2*acc*recall / (acc + recall)
            fpr, tpr, threshold = roc_curve(torch.tensor(torch.stack(pred_list, 0)), torch.tensor(torch.stack(label_list, 0)))
            
            writer.add_figure("Confusion matrix", sns.heatmap(df_cm, annot=True, cmap='coolwarm_r').get_figure(), i_epoch)
            writer.add_scalar(f'Loss/Train loss', train_loss, global_step=i_epoch)
            writer.add_scalar(f'Loss/Val loss', val_loss, global_step=i_epoch)
            writer.add_scalar(f'Metrics/Accuracy', acc, global_step=i_epoch)
            writer.add_scalar(f'Metrics/Recall', recall, global_step=i_epoch)
            writer.add_scalar(f'Metrics/F1', f1, global_step=i_epoch)

            fig1 = plt.figure(figsize=(15,8)) 
            ax = fig1.subplots()
            for i,w,k in zip(range(cfg.n_classes), classes, 'bgrcm'):
                ax.plot(fpr[i].cpu().detach().numpy(),tpr[i].cpu().detach().numpy(), c=k, label=w)
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curve')
                ax.legend(loc='lower right') 
            plt.show()
            writer.add_figure(f'Metrics/ROC_curve', fig1, global_step=i_epoch)
            
            tqdm.write(f'[VAL Epoch {i_epoch}]: Validation Loss: {val_loss:.8f}, Accuracy: {acc:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

        if i_epoch % cfg.i_save == 0:
            save_path = path.join(log_path, f'{i_epoch}.pth')
            torch.save(network.state_dict(), save_path)
            
            logits = []
            ys = []
            with torch.no_grad():
                for i, data in enumerate(heart_valloader):
                    inputs, labels = data
                    inputs = inputs.to(cfg.device)
                    labels = labels.long().to(cfg.device)
                    inputs = encoder(inputs.float())
                    outputs = network(inputs)
                    logits.append(torch.softmax(outputs, dim=-1))
                    ys.append(labels)
            logits = torch.cat(logits, dim=0).detach().cpu().numpy()
            ys = torch.cat(ys, dim=0).detach().cpu().numpy()
            logits = np.concatenate([logits, ys[..., None]], axis=-1)
            np.save(path.join(log_path, f'logits_{i_epoch}.npy'), logits)

        if i_epoch % cfg.i_pred == 0:      
            heart_testing = HeartBeatDataset(np.asarray(test_df), y=None)
            heart_testloader = DataLoader(heart_testing, batch_size=cfg.n_batches, shuffle=False)
            outputs = []
            # preds = []
            with torch.no_grad():
                for i, data in enumerate(heart_testloader):
                    inputs = data
                    inputs = inputs.to(cfg.device)
                    inputs = encoder(inputs.float())
                    output = network(inputs)
                    # pred = torch.argmax(outputs, dim=-1)
                    # preds.append(pred)
                    outputs.append(output)
            # preds = torch.cat(preds, dim=0).cpu().detach().numpy()
            outputs = torch.cat(outputs, dim=0).cpu().detach().numpy()
            results = pd.DataFrame(data=outputs, columns=['N', 'S', 'V', 'N+V', 'U'])
            results['preds'] = np.argmax(outputs, axis=1)
            results['mod_preds'] = np.argmax(outputs, axis=1)
            
            results.loc[(results['S']>0.05) & (results['preds']==0), 'modified_pred'] = 1
            results.loc[(results['V']>0.05) & (results['preds']==0), 'modified_pred'] = 2
            results.loc[(results['N+V']>0.05) & (results['preds']==0), 'modified_pred'] = 3
            
            results['id'] = [i for i in range(1, results.shape[0]+1)]  
            results[['id', 'preds']].to_csv(path.join(log_path, f'{i_epoch}.csv'), index=False)
            results[['id', 'mod_preds']].to_csv(path.join(log_path, f'mod_{i_epoch}.csv'), index=False)