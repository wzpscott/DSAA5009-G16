from torch.utils.data import Dataset

class HeartBeatDataset(Dataset):
    def __init__(self, X_train, X_test, y_train, y_test, mode='train'):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
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
            return self.X_train[idx], self.y_train[idx]
        elif self.mode == 'test':
            return self.X_test[idx], self.y_test[idx]
        else:
            raise ValueError(f'Wrong mode: {self.mode}')
        
        