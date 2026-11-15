"""Task-pre-demo: tensors, datasets & dataloaders, transforms, building a neural network.

Run:  python pre_demo.py     (press Enter to move to the next section)
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Normalize, Compose

DATA_DIR = "../fashion-mnist/data"  # reuses the Fashion MNIST files already downloaded


def section(title):
    input(f"\n[Enter] ")
    print("=" * 60)
    print(title)
    print("=" * 60)


# ---------------------------------------------------------------- 1. Tensors
section("1. TENSORS")
data = [[1, 2], [3, 4]]
x = torch.tensor(data)                      # from Python data
print("from list     :", x)
print("zeros / ones  :", torch.zeros(2, 3).shape, torch.ones(2, 3).dtype)
print("random        :", torch.rand(2, 3))
print("shape, dtype, device:", x.shape, x.dtype, x.device)

a = torch.rand(2, 3)
b = torch.rand(3, 2)
print("matrix multiply a @ b ->", (a @ b).shape)
print("element-wise a * a   ->", (a * a).shape)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("moving a tensor to the accelerator:", a.to(device).device)

# ------------------------------------------------------------ 2. Datasets
section("2. DATASETS (Fashion MNIST)")
train_data = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=ToTensor())
print("train samples:", len(train_data), " test samples:", len(test_data))
image, label = train_data[0]
print("one sample    : image", tuple(image.shape), "label", label, "=", train_data.classes[label])
print("classes       :", train_data.classes)

# ------------------------------------------------------------ 3. Transforms
section("3. TRANSFORMS")
raw = datasets.FashionMNIST(DATA_DIR, train=True)           # no transform: PIL image
pil_image, _ = raw[0]
print("without transform :", type(pil_image).__name__, pil_image.size)

to_tensor = ToTensor()
t = to_tensor(pil_image)
print("ToTensor          :", t.shape, "values in", (t.min().item(), t.max().item()))

normalize = Normalize((0.5,), (0.5,))
n = normalize(t)
print("Normalize(0.5,0.5):", "values in", (n.min().item(), n.max().item()))

pipeline = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
print("Compose([...])    :", pipeline(pil_image).shape)
print("(the CIFAR10 code uses the same idea with 3-channel mean/std values)")

# ----------------------------------------------------------- 4. DataLoader
section("4. DATALOADERS")
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)
X, y = next(iter(train_loader))
print("one batch X:", tuple(X.shape), "(batch, channels, height, width)")
print("one batch y:", tuple(y.shape), y[:8].tolist())
print("batches per epoch:", len(train_loader))

# ---------------------------------------------------------- 5. Build the NN
section("5. BUILDING THE NEURAL NETWORK")


class NeuralNetwork(nn.Module):
    def __init__(self, hidden_size=512):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(28 * 28, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 10),
        )

    def forward(self, x):
        return self.stack(self.flatten(x))


model = NeuralNetwork().to(device)
print(model)
print("parameters:", sum(p.numel() for p in model.parameters()))

logits = model(X[:1].to(device))
probs = nn.Softmax(dim=1)(logits)
print("output logits shape:", tuple(logits.shape))
print("predicted class (untrained):", train_data.classes[probs.argmax(1).item()])

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
print("loss function:", loss_fn)
print("optimizer    :", type(optimizer).__name__)

input("\n[Enter] to finish")
print("Done: tensors -> datasets -> transforms -> dataloaders -> model.")
