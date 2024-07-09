import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

def train_model(model, dataset, learning_rate, batch_size, epochs):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        epoch_loss = 0.0
        for batch in data_loader:
            inputs, labels = batch['X'], batch['y']
            inputs = torch.tensor(inputs, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.float32)
            model.zero_grad()
            outputs = model(inputs)
            mse_loss = nn.MSELoss()
            loss = mse_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(data_loader.dataset)
        print(f'Epoch {epoch+1}/{epochs} - Loss: {epoch_loss}')