from torch.utils.data import Dataset
import numpy as np

class HeartBeatDataset(Dataset):
    def __init__(self, X_train, X_test, y_train, y_test, mode='train'):
        self.X_train = np.asarray(X_train)
        self.X_test = X_test
        self.y_train = np.asarray(y_train)
        self.y_test = y_test
        
        self.mode = mode
        
    def train(self):
        self.mode = 'train'
        
    def test(self):
        self.mode = 'test'
        
    def __len__(self):
        if self.mode == 'train':
            return self.X_train.shape[0]
        elif self.mode == 'test':
            return self.X_test.shape[0]
        else:
            raise ValueError(f'Wrong mode: {self.mode}')
        
    def __getitem__(self, idx):
        if self.mode == 'train':
            print(self.X_train.shape, self.y_train.shape)
            return self.X_train[idx], self.y_train[idx]
        elif self.mode == 'test':
            return self.X_test[idx], self.y_test[idx]
        else:
            raise ValueError(f'Wrong mode: {self.mode}')
        

if __name__ == '__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split
    train_df = pd.read_csv('./data/train.csv')
    X = train_df.iloc[:, 1: -1]
    y = train_df.iloc[:, -1]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X.shape, y.shape, X_train.shape, X_val.shape, y_train.shape, y_val.shape)
    
    dataset = HeartBeatDataset(X_train, X_val, y_train, y_val)
    print(dataset[0])