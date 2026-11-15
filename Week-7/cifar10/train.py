"""CIFAR10 image classifier: a small convolutional neural network (CNN).

Usage:
    python train.py                                   # baseline model -> base/model.pth
    python train.py --optimizer adam --lr 1e-3 --epochs 15 --width 64 --out optimized/model.pth

The functions here are reused by tune.py (hyperparameter search).
"""
import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, ToTensor

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Per-channel mean and std of the CIFAR10 training set (used to normalize the pixels).
# predict.py must use exactly the same numbers.
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


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
    """Download CIFAR10 and wrap it in DataLoaders.

    Transforms: ToTensor() turns a PIL image (32x32 RGB, 0-255) into a float tensor
    [3, 32, 32] in 0-1, then Normalize() centers every color channel around zero.
    """
    transform = Compose([ToTensor(), Normalize(MEAN, STD)])
    train_data = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    return train_loader, test_loader


# ------------------------------------------------ 3) The neural network
class CNN(nn.Module):
    """3 convolution blocks (each: 2 convolutions + pooling) followed by one linear layer.

    `width` is the number of filters of the first block; the next blocks use 2x and 4x.
    Images shrink 32 -> 16 -> 8 -> 4 pixels while the number of feature maps grows.
    """

    def __init__(self, width: int = 32):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, width), block(width, 2 * width), block(2 * width, 4 * width))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),                    # randomly switches off neurons against overfitting
            nn.Linear(4 * width * 4 * 4, 10),   # raw scores (logits), one per class
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ------------------------------------------- 4) Training and evaluation
def make_optimizer(name: str, params, lr: float):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    return torch.optim.SGD(params, lr=lr)  # "sgd": plain gradient descent (the Quickstart default)


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


def train_model(lr=1e-3, batch_size=64, epochs=5, width=16, optimizer="sgd", device=None,
                loaders=None, verbose=True):
    """Train a model and return (model, test_accuracy). tune.py calls this in a loop."""
    device = device or get_device()
    train_loader, test_loader = loaders or get_dataloaders(batch_size)

    model = CNN(width).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = make_optimizer(optimizer, model.parameters(), lr)

    acc = 0.0
    for epoch in range(1, epochs + 1):
        train_one_epoch(train_loader, model, loss_fn, opt, device)
        loss, acc = evaluate(test_loader, model, loss_fn, device)
        if verbose:
            print(f"Epoch {epoch:2d}/{epochs}  test loss {loss:.4f}  test accuracy {acc:.2%}")
    return model, acc


# -------------------------------------------------------- 5) Save the model
def save_model(model, width: int, path: str):
    """Save weights on the CPU so the Cog container (CPU only) can load them."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save({"width": width, "state_dict": state}, path)
    print(f"Saved model to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train a CIFAR10 classifier")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--optimizer", choices=["sgd", "momentum", "adam"], default="sgd")
    parser.add_argument("--out", default="base/model.pth")
    args = parser.parse_args()

    device = get_device()
    print(f"Using {device} device")
    model, acc = train_model(args.lr, args.batch_size, args.epochs, args.width, args.optimizer, device)
    print(f"Final test accuracy: {acc:.2%}")
    save_model(model, args.width, args.out)


if __name__ == "__main__":
    main()
