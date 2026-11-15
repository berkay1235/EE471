# Week 7: Image Classifiers, Hyperparameter Optimization and Cog Containers

Two image classifiers (Fashion MNIST and CIFAR10), each trained in two versions
(baseline and hyperparameter-optimized with Optuna), packaged as Cog/Docker
containers and compared side by side on a small Flask website.

| Task | Dataset | Model | Baseline accuracy | Optimized accuracy |
|------|---------|-------|-------------------|--------------------|
| 1 | Fashion MNIST | MLP (784 → h → h → 10) | 64.91% | 88.97% |
| 2 | CIFAR10 | 3-block CNN with BatchNorm | 56.30% | 83.65% |

## Folder layout

```
Week-7/
├── fashion-mnist/
│   ├── train.py              # PyTorch training script (CLI arguments)
│   ├── tune.py               # Optuna hyperparameter search
│   ├── tuning_results.csv    # every trial of the search
│   ├── base/                 # baseline model + Cog predictor (model.pth, predict.py, cog.yaml)
│   ├── optimized/            # optimized model + Cog predictor
│   └── website/              # Flask comparison site
└── cifar10/                  # same structure as fashion-mnist
```

## Requirements

- Python 3.11+ (developed with 3.14), Docker, and [Cog](https://github.com/replicate/cog)
- Trained models (`model.pth`) are committed, so training is optional.

Create one virtual environment for the training scripts and the website:

```bash
cd Week-7/fashion-mnist
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision optuna
pip install -r website/requirements.txt
```

## Ports used

| Service | Port |
|---------|------|
| Fashion baseline container | 5101 |
| Fashion optimized container | 5102 |
| CIFAR10 baseline container | 5111 |
| CIFAR10 optimized container | 5112 |
| Fashion website | 5200 |
| CIFAR10 website | 5300 |

## Task 1: Fashion MNIST

### Train (optional, models are already included)

```bash
cd Week-7/fashion-mnist
python train.py                      # baseline  -> base/model.pth
python tune.py                       # Optuna search -> tuning_results.csv
python train.py --lr 0.0638 --batch-size 32 --hidden-size 1024 --epochs 15 \
                --out optimized/model.pth
```

### Build and run the containers

```bash
cd base
cog build -t fashion-base
docker run -d --rm --name fashion-base -p 5101:5000 fashion-base

cd ../optimized
cog build -t fashion-optimized
docker run -d --rm --name fashion-optimized -p 5102:5000 fashion-optimized
```

### Test with curl

```bash
curl http://localhost:5101/health-check
curl -s -X POST http://localhost:5101/predictions \
  -H "Content-Type: application/json" \
  -d "{\"input\":{\"image\":\"data:image/png;base64,$(base64 -i base/sample.png)\"}}"
```

### Watch the prediction log inside the container

```bash
docker exec -it fashion-base tail -F /tmp/predictions.log
```

### Run the website

```bash
cd Week-7/fashion-mnist
COG_URL_BASE=http://localhost:5101/predictions \
COG_URL_OPT=http://localhost:5102/predictions \
python website/app.py
```

Open http://localhost:5200 and upload a garment image. The baseline and the
optimized model answer side by side.

## Task 2: CIFAR10

### Train (optional)

```bash
cd Week-7/cifar10
python train.py                      # baseline  -> base/model.pth
python tune.py                       # Optuna search -> optimized/model.pth + tuning_results.csv
```

The first run downloads CIFAR10 to `data/` (about 163 MB). If the download
server resets the connection, fetch the archive with
`curl -L --retry 8 --retry-all-errors -C - -o data/cifar-10-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`
and run `train.py` again.

### Build and run the containers

```bash
cd base
cog build -t cifar-base
docker run -d --rm --name cifar-base -p 5111:5000 cifar-base

cd ../optimized
cog build -t cifar-optimized
docker run -d --rm --name cifar-optimized -p 5112:5000 cifar-optimized
```

### Test with curl and watch the log

```bash
curl -s -X POST http://localhost:5111/predictions \
  -H "Content-Type: application/json" \
  -d "{\"input\":{\"image\":\"data:image/png;base64,$(base64 -i base/sample.png)\"}}"
docker exec -it cifar-base tail -F /tmp/predictions.log
```

### Run the website

```bash
cd Week-7/cifar10
COG_URL_BASE=http://localhost:5111/predictions \
COG_URL_OPT=http://localhost:5112/predictions \
python website/app.py
```

Open http://localhost:5300.

## Postman

Send a `POST` request to `http://localhost:5111/predictions` (or 5101, 5102,
5112) with a JSON body:

```json
{ "input": { "image": "data:image/png;base64,<base64 of the image>" } }
```

The response contains `predicted_class`, `confidence` and the `top5` list.

## Hyperparameter optimization

Optuna (TPE sampler, seed 42) maximizes the test accuracy.

| | Fashion MNIST | CIFAR10 |
|---|---|---|
| Search space | lr 1e-3 to 1.0 (log), batch size {32, 64, 128, 256}, hidden size {128, 256, 512, 1024}, epochs {5, 10, 15} | optimizer {sgd, momentum, adam}, lr 1e-4 to 0.3 (log), batch size {64, 128, 256}, width {16, 32, 64}, epochs {5, 10, 15} |
| Trials | 15 | 12 |
| Baseline | SGD, lr 0.001, batch 64, hidden 512, 5 epochs | SGD, lr 0.001, batch 64, width 16, 5 epochs |
| Best | lr 0.0638, batch 32, hidden 1024, 15 epochs | Adam, lr 0.00397, batch 64, width 64, 10 epochs |

All trials are listed in each folder's `tuning_results.csv`.
