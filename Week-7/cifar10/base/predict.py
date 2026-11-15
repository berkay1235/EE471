from datetime import datetime

from cog import BasePredictor, Input, Path
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision.transforms import Compose, Normalize, ToTensor

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Same normalization numbers as in train.py
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

# Prediction log inside the container (watch it with: docker exec -it <name> tail -F /tmp/predictions.log)
LOG_FILE = "/tmp/predictions.log"


class CNN(nn.Module):
    """Same architecture as in train.py (the weights only fit this exact layout)."""

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
            nn.Dropout(0.3),
            nn.Linear(4 * width * 4 * 4, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


TRANSFORM = Compose([ToTensor(), Normalize(MEAN, STD)])


def preprocess(img: Image.Image) -> torch.Tensor:
    """Turn any photo into a CIFAR10 style input: 32x32 RGB, normalized, batch of one."""
    img = ImageOps.fit(img.convert("RGB"), (32, 32))  # center crop + resize
    return TRANSFORM(img).unsqueeze(0)  # [1, 3, 32, 32]


class Predictor(BasePredictor):
    def setup(self):
        """Load the trained weights once, when the container starts."""
        checkpoint = torch.load("model.pth", map_location="cpu", weights_only=True)
        self.model = CNN(checkpoint["width"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        print("Model loaded, ready for predictions", flush=True)

    def predict(self, image: Path = Input(description="Photo of an airplane, car, animal, ship or truck")) -> dict:
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
