import torch
from sklearn.linear_model import LinearRegression
from datasets import load_diabetes_dataset, BasicDataset
from models import MyLinear
from training import train_model
from evaluation import evaluate_model

# Hyperparameters
batch_size = 64
learning_rate = 0.01
epochs = 25
in_features = 10
out_features = 1

# Load dataset and initialize models
X_train, X_test, y_train, y_test = load_diabetes_dataset()
dataset_train = BasicDataset(X_train, y_train)
my_linear = MyLinear(in_features, out_features, bias=False)
pt_linear = torch.nn.Linear(in_features, out_features)

# Train and evaluate MyLinear model
print('MyLinear')
train_model(my_linear, dataset_train, learning_rate, batch_size, epochs)
evaluate_model(my_linear, X_test, y_test)
print()

# Train and evaluate torch.nn.Linear model
print('torch.nn.Linear')
train_model(pt_linear, dataset_train, learning_rate, batch_size, epochs)
evaluate_model(pt_linear, X_test, y_test)
print()

# Evaluate sklearn linear model
print('sklearn.linear_model.LinearRegression')
print('Test MSE:', np.mean(np.power(LinearRegression().fit(X_train, y_train).predict(X_test) - y_test, 2)))