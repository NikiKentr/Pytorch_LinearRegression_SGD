import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_diabetes_dataset():
    data = load_diabetes()
    X, y = data.data, data.target
    scaler_X = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y.reshape(-1, 1))
    X = scaler_X.transform(X)
    y = scaler_y.transform(y.reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2021)
    return X_train, X_test, y_train, y_test

class BasicDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        return dict(X=self.X[idx], y=self.y[idx])

    def __len__(self):
        return len(self.X)