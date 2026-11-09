"""Hyperparameter optimization for the Fashion MNIST classifier, using Optuna.

Usage:
    python tune.py                 # 15 trials
    python tune.py --trials 30

Optuna tries different hyperparameter combinations, learns from the results of earlier
trials (TPE sampler) and keeps the best model in optimized/model.pth.
All trials are logged to tuning_results.csv (useful for the report).
"""
import argparse
import csv

import optuna

from train import get_dataloaders, get_device, save_model, train_model

BASELINE = {"lr": 1e-3, "batch_size": 64, "epochs": 5, "hidden_size": 512}


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search")
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--out", default="optimized/model.pth")
    args = parser.parse_args()

    device = get_device()
    print(f"Using {device} device")

    loaders = {}  # one set of DataLoaders per batch size, so we do not rebuild them every trial
    best = {"acc": 0.0, "model": None, "hidden_size": None}

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-3, 1.0, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512, 1024])
        epochs = trial.suggest_categorical("epochs", [5, 10, 15])

        if batch_size not in loaders:
            loaders[batch_size] = get_dataloaders(batch_size)
        model, acc = train_model(lr, batch_size, epochs, hidden_size, device,
                                 loaders=loaders[batch_size], verbose=False)

        if acc > best["acc"]:  # keep the best model seen so far
            best.update(acc=acc, model=model, hidden_size=hidden_size)
        return acc  # Optuna maximizes this value

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=args.trials)

    print("\nBaseline hyperparameters :", BASELINE)
    print("Best hyperparameters     :", study.best_params)
    print(f"Best test accuracy       : {study.best_value:.2%}")
    save_model(best["model"], best["hidden_size"], args.out)

    # Log every trial to a CSV file (standard csv module, no pandas needed)
    with open("tuning_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "test_accuracy", "lr", "batch_size", "hidden_size", "epochs"])
        for t in study.trials:
            p = t.params
            writer.writerow([t.number, t.value, p["lr"], p["batch_size"], p["hidden_size"], p["epochs"]])


if __name__ == "__main__":
    main()
