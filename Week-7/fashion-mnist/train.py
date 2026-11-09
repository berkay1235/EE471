"""Fashion MNIST garment classifier (based on the PyTorch Quickstart tutorial).

Usage:
    python train.py                                  # baseline model -> base/model.pth
    python train.py --lr 5e-3 --epochs 15 --out optimized/model.pth

The functions here are reused by tune.py (hyperparameter search).
"""
import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


# ---------------------------------------------------------------- 1) Device
def get_device() -> str:
    """Use the Apple GPU (mps) or CUDA if available, otherwise the CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ------------------------------------------- 2) Datasets and DataLoaders
def get_dataloaders(batch_size: int, data_dir: str = "data"):
    """Download Fashion MNIST and wrap it in DataLoaders.

    ToTensor() is the transform: PIL image (28x28, 0-255) -> float tensor [1, 28, 28] in 0-1.
    """
    train_data = datasets.FashionMNIST(data_dir, train=True, download=True, transform=ToTensor())
    test_data = datasets.FashionMNIST(data_dir, train=False, download=True, transform=ToTensor())
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    return train_loader, test_loader


# ------------------------------------------------ 3) The neural network
class NeuralNetwork(nn.Module):
    """Fully connected network: 784 pixels -> hidden -> hidden -> 10 classes."""

    def __init__(self, hidden_size: int = 512):
        super().__init__()
        self.flatten = nn.Flatten()  # [N, 1, 28, 28] -> [N, 784]
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10),  # raw scores (logits), one per class
        )

    def forward(self, x):
        return self.layers(self.flatten(x))


# ------------------------------------------- 4) Training and evaluation
def train_one_epoch(loader, model, loss_fn, optimizer, device):
    model.train()
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        loss = loss_fn(model(X), y)  # forward pass + loss
        loss.backward()              # backpropagation: compute gradients
        optimizer.step()             # update the weights
        optimizer.zero_grad()        # reset gradients for the next batch


def evaluate(loader, model, loss_fn, device):
    """Return (average loss, accuracy) over a dataset, without computing gradients."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += loss_fn(pred, y).item() * len(y)
            correct += (pred.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def train_model(lr=1e-3, batch_size=64, epochs=5, hidden_size=512, device=None,
                loaders=None, verbose=True):
    """Train a model and return (model, test_accuracy). tune.py calls this in a loop."""
    device = device or get_device()
    train_loader, test_loader = loaders or get_dataloaders(batch_size)

    model = NeuralNetwork(hidden_size).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    acc = 0.0
    for epoch in range(1, epochs + 1):
        train_one_epoch(train_loader, model, loss_fn, optimizer, device)
        loss, acc = evaluate(test_loader, model, loss_fn, device)
        if verbose:
            print(f"Epoch {epoch:2d}/{epochs}  test loss {loss:.4f}  test accuracy {acc:.2%}")
    return model, acc


# -------------------------------------------------------- 5) Save the model
def save_model(model, hidden_size: int, path: str):
    """Save weights on the CPU so the Cog container (CPU only) can load them."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save({"hidden_size": hidden_size, "state_dict": state}, path)
    print(f"Saved model to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train a Fashion MNIST classifier")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--out", default="base/model.pth")
    args = parser.parse_args()

    device = get_device()
    print(f"Using {device} device")
    model, acc = train_model(args.lr, args.batch_size, args.epochs, args.hidden_size, device)
    print(f"Final test accuracy: {acc:.2%}")
    save_model(model, args.hidden_size, args.out)


if __name__ == "__main__":
    main()
