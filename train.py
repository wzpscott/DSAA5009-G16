from networks import *
from encoders import *
from dataset import *
import torch
import torch.nn as nn
import os
import pandas as pd
import torch.optim as optim
from tqdm import tqdm, trange
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import metrics as ms
from torch.utils.data import Dataset, DataLoader

train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

target_col = '187'
features_col = [x for x in train_df.columns if x != '187']

# Fixing a type for pytorch
X = np.asarray(train_df[features_col])
y = np.asarray(train_df[target_col])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(X.shape, y.shape, X_train.shape, X_val.shape, y_train.shape, y_val.shape)

# Train and validation
# selecting the number of epochs
num_epochs = 200
d_input = 1
d_hidden = 256
n_classes = 5
n_layers = 2
n_freqs = 8 # 16
bidirectional = False
n_batch = 1024
device = 'cuda:0'

# instantiating the dataset and dataloader for both training and validation
heart_training = HeartBeatDataset(X_train, X_val, y_train, y_val, 'train')
heart_validating = HeartBeatDataset(X_train, X_val, y_train, y_val, 'test')

heart_trainloader = DataLoader(heart_training, batch_size=n_batch, shuffle=True)
heart_valloader = DataLoader(heart_validating, batch_size=n_batch, shuffle=True)

# instantiating the model, loss function, and optimizer
encoder = PositionalEncoder(d_input, n_freqs).to(device)
model = LSTM(encoder.d_output, d_hidden, n_classes, n_layers, bidirectional).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# to store the loss
loss_list = list()

# training the model
for epoch in trange(num_epochs):
    model.train()
    running_loss = 0.0
    
    for i, data in enumerate(heart_trainloader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.long().to(device)
        optimizer.zero_grad()
        # print("input:", inputs.shape)
        inputs = encoder(inputs.float())
        outputs = model(inputs)
        # print("output:", outputs.shape, "labels:", labels.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    model.eval()
        
    # initialize the validation loss
    val_loss = 0.0
    results = 0.0
    # Disable gradient calculation
    with torch.no_grad():
        # Iterate over the validation data
        for i, data in enumerate(heart_valloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.long().to(device)
            inputs = encoder(inputs.float())
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            results += (torch.argmax(outputs, 1) == labels).float().mean()
    train_loss = running_loss/len(heart_training)
    val_loss = val_loss/len(heart_validating)
    results = results / (i + 1)
    loss_list.append((train_loss, val_loss))
    
    str_epoch = f'Epoch [{epoch+1}/{num_epochs}]'
    str_train_loss = f'Training Loss: {train_loss:.8f}'
    str_val_loss = f'Validation Loss: {val_loss:.8f}'
    results_value = f'Accuracy: {results:.8f}'
    print(f'{str_epoch}, {str_train_loss}, {str_val_loss}, {results_value}')

