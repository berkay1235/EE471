from datetime import datetime

from cog import BasePredictor, Input, Path
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision.transforms import ToTensor

CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# Prediction log inside the container (watch it with: docker exec -it <name> tail -F /tmp/predictions.log)
LOG_FILE = "/tmp/predictions.log"


class NeuralNetwork(nn.Module):
    """Same architecture as in train.py (the weights only fit this exact layout)."""

    def __init__(self, hidden_size: int = 512):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10),
        )

    def forward(self, x):
        return self.layers(self.flatten(x))


def preprocess(img: Image.Image) -> torch.Tensor:
    """Turn any photo into a Fashion MNIST style input: 28x28, grayscale, light item on dark background."""
    img = ImageOps.fit(img.convert("L"), (28, 28))  # grayscale + center crop + resize
    tensor = ToTensor()(img)  # [1, 28, 28], values 0-1
    # Fashion MNIST items are bright on a black background. Photos usually have a bright
    # background, so if the image border is bright we invert the colors.
    border = torch.cat([tensor[0, 0], tensor[0, -1], tensor[0, :, 0], tensor[0, :, -1]])
    if border.mean() > 0.5:
        tensor = 1.0 - tensor
    return tensor.unsqueeze(0)  # [1, 1, 28, 28] (batch of one)


class Predictor(BasePredictor):
    def setup(self):
        """Load the trained weights once, when the container starts."""
        checkpoint = torch.load("model.pth", map_location="cpu")
        self.model = NeuralNetwork(checkpoint["hidden_size"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        print("Model loaded, ready for predictions", flush=True)

    def predict(self, image: Path = Input(description="Photo of a garment")) -> dict:
        """Classify one image and return the best class plus the top 5."""
        batch = preprocess(Image.open(image))

        with torch.no_grad():
            probs = self.model(batch).softmax(dim=1)[0]

        top = probs.topk(5)
        top5 = [
            {"class": CLASSES[i], "confidence": round(p, 4)}
            for p, i in zip(top.values.tolist(), top.indices.tolist())
        ]

        line = f"{datetime.now().isoformat(timespec='seconds')} prediction: {top5[0]['class']} ({top5[0]['confidence']:.2%})"
        print(line, flush=True)
        with open(LOG_FILE, "a") as log:
            log.write(line + "\n")

        return {
            "predicted_class": top5[0]["class"],
            "confidence": top5[0]["confidence"],
            "top5": top5,
        }
