import numpy as np
from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

from config import get_config
from model import VisualTransformer

np.random.seed(0)
torch.manual_seed(0)

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        x, y = batch
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() / len(train_loader)

    return train_loss

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).item()
            total += len(x)

    accuracy = correct / total * 100
    return test_loss, accuracy


def main(config):
    # Loading data
    transform = ToTensor()

    train_set = MNIST(root='./../datasets', train=True, download=True, transform=transform)
    test_set = MNIST(root='./../datasets', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=128)

    # Defining model and training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    model = VisualTransformer((1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10).to(device)
    N_EPOCHS = 5
    LR = 0.005

    # Training loop
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    for epoch in trange(N_EPOCHS, desc="Training"):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

    # Test loop
    test_loss, accuracy = test(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.2f}")
    print(f"Test accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    main(config)