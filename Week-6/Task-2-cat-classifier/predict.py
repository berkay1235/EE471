from datetime import datetime

from cog import BasePredictor, Input, Path
import torch
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights

# ImageNet class ids of the domestic cat kinds
CAT_CLASS_IDS = {281, 282, 283, 284, 285}  # tabby, tiger cat, Persian, Siamese, Egyptian cat

# Prediction log inside the container (watch it with: docker exec -it cat-classifier tail -F /tmp/predictions.log)
LOG_FILE = "/tmp/predictions.log"


class Predictor(BasePredictor):
    def setup(self):
        """Load the pre-trained ResNet50 once, when the container starts."""
        weights = ResNet50_Weights.IMAGENET1K_V2
        self.model = resnet50(weights=weights).eval()
        self.preprocess = weights.transforms()
        self.labels = weights.meta["categories"]
        print("Model loaded, ready for predictions", flush=True)

    def predict(self, image: Path = Input(description="Photo of a cat")) -> dict:
        """Classify one image and return the best class plus the top 5."""
        img = Image.open(image).convert("RGB")
        batch = self.preprocess(img).unsqueeze(0)

        with torch.no_grad():
            probs = self.model(batch).softmax(dim=1)[0]

        top = probs.topk(5)
        top5 = [
            {"class": self.labels[i], "confidence": round(p, 4)}
            for p, i in zip(top.values.tolist(), top.indices.tolist())
        ]
        best_id = top.indices[0].item()

        line = f"{datetime.now().isoformat(timespec='seconds')} prediction: {top5[0]['class']} ({top5[0]['confidence']:.2%})"
        print(line, flush=True)
        with open(LOG_FILE, "a") as log:
            log.write(line + "\n")

        return {
            "predicted_class": top5[0]["class"],
            "confidence": top5[0]["confidence"],
            "is_cat": best_id in CAT_CLASS_IDS,
            "top5": top5,
        }