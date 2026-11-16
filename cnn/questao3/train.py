import argparse
import copy
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset

from cnn.questao3.data import load_cifar10
from cnn.questao3.models import CIFAR10CNN, CIFAR10MLP


CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
OUTPUT_DIR = Path(__file__).resolve().parent


def make_loaders(batch_size: int, seed: int, device: torch.device):
    train_cnn, test_cnn, train_mlp, test_mlp = load_cifar10()
    train_indices, val_indices = train_test_split(
        np.arange(len(train_cnn)),
        test_size=0.1,
        random_state=seed,
        stratify=train_cnn.targets,
    )

    def loaders(dataset, test_dataset):
        return (
            DataLoader(
                Subset(dataset, train_indices.tolist()),
                batch_size=batch_size,
                shuffle=True,
                pin_memory=device.type == "cuda",
            ),
            DataLoader(
                Subset(dataset, val_indices.tolist()),
                batch_size=batch_size,
                shuffle=False,
                pin_memory=device.type == "cuda",
            ),
            DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=device.type == "cuda",
            ),
        )

    cnn_loaders = loaders(train_cnn, test_cnn)
    mlp_loaders = loaders(train_mlp, test_mlp)
    return cnn_loaders, mlp_loaders


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    targets_all = []
    predictions_all = []

    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(inputs)
            total_loss += criterion(logits, targets).item() * targets.size(0)
            predictions = logits.argmax(dim=1)
            targets_all.extend(targets.cpu().tolist())
            predictions_all.extend(predictions.cpu().tolist())

    return (
        total_loss / len(loader.dataset),
        accuracy_score(targets_all, predictions_all),
        targets_all,
        predictions_all,
    )


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int,
    checkpoint_path: Path | None = None,
):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        val_loss, _, _, _ = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(
            f"Epoch {epoch:02d}/{epochs} "
            f"- train loss: {train_loss:.4f} - val loss: {val_loss:.4f}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_path is not None:
                torch.save(
                    {
                        "architecture": "CIFAR10CNN",
                        "model_state_dict": best_state,
                        "class_names": CLASS_NAMES,
                        "epoch": best_epoch,
                        "val_loss": best_val_loss,
                    },
                    checkpoint_path,
                )

    model.load_state_dict(best_state)
    return model, history, best_epoch, best_val_loss


def save_loss_plot(histories, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for axis, (name, history) in zip(axes, histories.items()):
        epochs = range(1, len(history["train_loss"]) + 1)
        axis.plot(epochs, history["train_loss"], label="Treino")
        axis.plot(epochs, history["val_loss"], label="Validação")
        axis.set(title=name, xlabel="Época", ylabel="Cross-Entropy Loss")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_confusion_plot(matrix, output_path: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Classe predita",
        ylabel="Classe verdadeira",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
                fontsize=8,
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def train_and_evaluate(
    name,
    model,
    loaders,
    device,
    epochs: int,
    checkpoint_path: Path | None = None,
):
    train_loader, val_loader, test_loader = loaders
    model, history, best_epoch, best_val_loss = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs,
        checkpoint_path,
    )
    test_loss, test_accuracy, targets, predictions = evaluate(
        model, test_loader, nn.CrossEntropyLoss(), device
    )
    matrix = confusion_matrix(targets, predictions, labels=range(len(CLASS_NAMES)))
    print(
        f"{name}: best epoch {best_epoch}, test loss {test_loss:.4f}, "
        f"test accuracy {test_accuracy:.4%}",
        flush=True,
    )
    return history, {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "confusion_matrix": matrix.tolist(),
    }, matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Treina MLP e CNN no CIFAR-10.")
    parser.add_argument("--mlp-epochs", type=int, default=5)
    parser.add_argument("--cnn-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}", flush=True)

    results_dir = OUTPUT_DIR / "results"
    checkpoints_dir = OUTPUT_DIR / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    cnn_loaders, mlp_loaders = make_loaders(args.batch_size, args.seed, device)
    mlp_history, mlp_metrics, mlp_matrix = train_and_evaluate(
        "MLP",
        CIFAR10MLP(),
        mlp_loaders,
        device,
        args.mlp_epochs,
    )
    cnn_history, cnn_metrics, cnn_matrix = train_and_evaluate(
        "CNN",
        CIFAR10CNN(),
        cnn_loaders,
        device,
        args.cnn_epochs,
        checkpoints_dir / "cifar10_cnn.pt",
    )

    save_loss_plot(
        {"MLP": mlp_history, "CNN": cnn_history},
        results_dir / "loss_curves.png",
    )
    save_confusion_plot(
        mlp_matrix, results_dir / "mlp_confusion_matrix.png", "Matriz de confusão — MLP"
    )
    save_confusion_plot(
        cnn_matrix, results_dir / "cnn_confusion_matrix.png", "Matriz de confusão — CNN"
    )
    with (results_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset": "CIFAR-10",
                "validation_fraction": 0.1,
                "seed": args.seed,
                "mlp": {"history": mlp_history, **mlp_metrics},
                "cnn": {"history": cnn_history, **cnn_metrics},
            },
            file,
            indent=2,
        )


if __name__ == "__main__":
    main()
