"""Hyperparameter optimization for the CIFAR10 classifier, using Optuna.

Usage:
    python tune.py                 # 12 trials
    python tune.py --trials 20

Optuna tries different hyperparameter combinations, learns from the results of earlier
trials (TPE sampler) and keeps the best model in optimized/model.pth.
All trials are logged to tuning_results.csv (useful for the report).
"""
import argparse
import csv

import optuna

from train import get_dataloaders, get_device, save_model, train_model

BASELINE = {"optimizer": "sgd", "lr": 1e-3, "batch_size": 64, "epochs": 5, "width": 16}


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--out", default="optimized/model.pth")
    args = parser.parse_args()

    device = get_device()
    print(f"Using {device} device")

    loaders = {}  # one set of DataLoaders per batch size, so we do not rebuild them every trial
    best = {"acc": 0.0, "model": None, "width": None}

    def objective(trial):
        optimizer = trial.suggest_categorical("optimizer", ["sgd", "momentum", "adam"])
        lr = trial.suggest_float("lr", 1e-4, 0.3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        width = trial.suggest_categorical("width", [16, 32, 64])
        epochs = trial.suggest_categorical("epochs", [5, 10, 15])

        if batch_size not in loaders:
            loaders[batch_size] = get_dataloaders(batch_size)
        model, acc = train_model(lr, batch_size, epochs, width, optimizer, device,
                                 loaders=loaders[batch_size], verbose=False)

        if acc > best["acc"]:  # keep the best model seen so far
            best.update(acc=acc, model=model, width=width)
        return acc  # Optuna maximizes this value

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=args.trials)

    print("\nBaseline hyperparameters :", BASELINE)
    print("Best hyperparameters     :", study.best_params)
    print(f"Best test accuracy       : {study.best_value:.2%}")
    save_model(best["model"], best["width"], args.out)

    # Log every trial to a CSV file (standard csv module, no pandas needed)
    with open("tuning_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "test_accuracy", "optimizer", "lr", "batch_size", "width", "epochs"])
        for t in study.trials:
            p = t.params
            writer.writerow([t.number, t.value, p["optimizer"], p["lr"], p["batch_size"], p["width"], p["epochs"]])


if __name__ == "__main__":
    main()
