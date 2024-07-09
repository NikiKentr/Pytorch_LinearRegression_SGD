import numpy as np
import torch

def evaluate_model(model, X_test, y_test):
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).float()
    yhat = model.forward(X_test_tensor).detach().numpy()
    mse = np.mean(np.power(yhat - y_test, 2))
    print('Test MSE:', mse)