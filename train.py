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
import warnings

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

def test_model(model, test_loader, criterion, device):
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

    train_set = MNIST(root='./datasets', train = True, download = True, transform=transform)
    test_set = MNIST(root='./datasets', train = False, download = True, transform=transform)

    train_loader = DataLoader(train_set, shuffle = True, batch_size = 128)
    test_loader = DataLoader(test_set, shuffle = False, batch_size = 128)

    # Defining model and training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    model = VisualTransformer(chw = config['chw'], n_patches = config['n_patches'], n_blocks = config['n_blocks'], hidden_d = config['hidden_d'], n_heads = config['n_heads'], out_d = config['out_d']).to(device)
    N_EPOCHS = config['num_epochs']
    LR = config['lr']

    # Training loop
    optimizer = Adam(model.parameters(), lr = LR)
    criterion = CrossEntropyLoss()

    for epoch in trange(N_EPOCHS, desc = "Training"):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

    # Test loop
    test_loss, accuracy = test_model(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.2f}")
    print(f"Test accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    main(config)