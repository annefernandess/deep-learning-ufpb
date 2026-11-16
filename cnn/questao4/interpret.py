import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from cnn.questao3.data import load_cifar10
from cnn.questao3.models import CIFAR10CNN


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
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[1]
    / "questao3"
    / "checkpoints"
    / "cifar10_cnn.pt"
)
ACTIVATION_LAYER_INDEX = 6


def load_model(device: torch.device) -> CIFAR10CNN:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint da Questão 3 não encontrado: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )
    model = CIFAR10CNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_test_set(model, test_dataset, device: torch.device):
    loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    probabilities = []
    labels = []

    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device))
            probabilities.append(logits.softmax(dim=1).cpu())
            labels.append(targets)

    return torch.cat(probabilities).numpy(), torch.cat(labels).numpy()


def select_examples(probabilities: np.ndarray, labels: np.ndarray):
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct_mask = predictions == labels

    correct_indices = []
    for label in range(len(CLASS_NAMES)):
        candidates = np.flatnonzero(correct_mask & (labels == label))
        if candidates.size:
            best = candidates[np.argmax(confidence[candidates])]
            correct_indices.append(int(best))
        if len(correct_indices) == 5:
            break
    if len(correct_indices) < 5:
        remaining = np.flatnonzero(correct_mask)
        remaining = remaining[
            np.argsort(confidence[remaining])[::-1]
        ]
        for index in remaining:
            if int(index) not in correct_indices:
                correct_indices.append(int(index))
            if len(correct_indices) == 5:
                break

    incorrect_indices = np.flatnonzero(~correct_mask)
    incorrect_indices = incorrect_indices[
        np.argsort(confidence[incorrect_indices])[::-1][:5]
    ]
    return correct_indices, incorrect_indices.tolist(), predictions, confidence


def save_first_layer_filters(model: CIFAR10CNN, output_path: Path) -> None:
    weights = model.features[0].weight.detach().cpu().numpy()
    figure, axes = plt.subplots(4, 8, figsize=(12, 7))

    for index, axis in enumerate(axes.flat):
        kernel = weights[index].transpose(1, 2, 0)
        minimum = kernel.min()
        maximum = kernel.max()
        displayed = (kernel - minimum) / (maximum - minimum + 1e-8)
        axis.imshow(displayed)
        axis.set_title(f"Filtro {index}")
        axis.axis("off")

    figure.suptitle("Filtros RGB aprendidos pela primeira convolução")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_feature_maps(
    model: CIFAR10CNN,
    test_dataset,
    indices: list[int],
    probabilities: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    output_path: Path,
):
    images = torch.stack([test_dataset[index][0] for index in indices])
    captured = {}
    layer = model.features[ACTIVATION_LAYER_INDEX]
    handle = layer.register_forward_hook(
        lambda _module, _inputs, output: captured.update(
            activations=output.detach().cpu()
        )
    )
    try:
        with torch.inference_mode():
            model(images)
    finally:
        handle.remove()

    if "activations" not in captured:
        raise RuntimeError("O hook não capturou ativações da camada intermediária.")

    activations = captured["activations"]
    figure, axes = plt.subplots(len(indices), 5, figsize=(15, 3 * len(indices)))
    axes = np.atleast_2d(axes)
    rows = []

    for row, index in enumerate(indices):
        image = images[row].permute(1, 2, 0).numpy()
        feature_maps = activations[row].numpy()
        channel_scores = feature_maps.mean(axis=(1, 2))
        channels = np.argsort(channel_scores)[-4:][::-1]

        axes[row, 0].imshow(image)
        axes[row, 0].set_title(
            f"Índice {index}\nVerdadeira: {CLASS_NAMES[labels[index]]}"
        )
        axes[row, 0].axis("off")

        for column, channel in enumerate(channels, start=1):
            feature_map = torch.from_numpy(feature_maps[channel])[None, None]
            heatmap = functional.interpolate(
                feature_map,
                size=image.shape[:2],
                mode="bilinear",
                align_corners=False,
            )[0, 0].numpy()
            heatmap /= heatmap.max() + 1e-8
            axes[row, column].imshow(image)
            axes[row, column].imshow(
                heatmap,
                cmap="inferno",
                alpha=0.55,
                vmin=0,
                vmax=1,
            )
            axes[row, column].set_title(f"Canal {channel} (mapa de ativação)")
            axes[row, column].axis("off")

        rows.append(
            {
                "tipo": "mapa_de_ativacao",
                "indice_teste": index,
                "classe_verdadeira": CLASS_NAMES[labels[index]],
                "classe_predita": CLASS_NAMES[predictions[index]],
                "confianca_predita": f"{probabilities[index, predictions[index]]:.6f}",
                "canais_visualizados": ",".join(str(channel) for channel in channels),
            }
        )

    figure.suptitle(
        "Quatro maiores ativações médias da segunda camada convolucional",
        y=1.01,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return rows


def save_classification_examples(
    test_dataset,
    correct_indices: list[int],
    incorrect_indices: list[int],
    probabilities: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    output_path: Path,
):
    groups = (
        ("Corretas — maior confiança", correct_indices),
        ("Incorretas — maior confiança", incorrect_indices),
    )
    figure, axes = plt.subplots(2, 5, figsize=(18, 7))
    rows = []

    for row, (title, indices) in enumerate(groups):
        for column, index in enumerate(indices):
            image, _ = test_dataset[index]
            prediction = int(predictions[index])
            confidence = float(probabilities[index, prediction])
            axes[row, column].imshow(image.permute(1, 2, 0).numpy())
            axes[row, column].set_title(
                f"{title}\nVerdadeira: {CLASS_NAMES[labels[index]]}\n"
                f"Predita: {CLASS_NAMES[prediction]} ({confidence:.1%})"
            )
            axes[row, column].axis("off")
            rows.append(
                {
                    "tipo": "correta" if row == 0 else "incorreta",
                    "indice_teste": index,
                    "classe_verdadeira": CLASS_NAMES[labels[index]],
                    "classe_predita": CLASS_NAMES[prediction],
                    "confianca_predita": f"{confidence:.6f}",
                    "canais_visualizados": "",
                }
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    _, test_dataset, _, _ = load_cifar10()
    probabilities, labels = predict_test_set(model, test_dataset, device)
    correct_indices, incorrect_indices, predictions, confidence = select_examples(
        probabilities, labels
    )

    save_first_layer_filters(
        model, OUTPUT_DIR / "first_layer_filters.png"
    )
    feature_rows = save_feature_maps(
        model,
        test_dataset,
        correct_indices,
        probabilities,
        labels,
        predictions,
        OUTPUT_DIR / "test_feature_maps.png",
    )
    classification_rows = save_classification_examples(
        test_dataset,
        correct_indices,
        incorrect_indices,
        probabilities,
        labels,
        predictions,
        OUTPUT_DIR / "classification_examples.png",
    )

    with (OUTPUT_DIR / "selected_examples.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        fieldnames = [
            "tipo",
            "indice_teste",
            "classe_verdadeira",
            "classe_predita",
            "confianca_predita",
            "canais_visualizados",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(feature_rows + classification_rows)

    print(f"Dispositivo: {device}")
    print(f"Amostras corretas para mapas: {correct_indices}")
    print(f"Amostras incorretas de maior confiança: {incorrect_indices}")
    print(f"Acurácia no teste: {(predictions == labels).mean():.4%}")
    print(f"Maior confiança entre erros: {confidence[incorrect_indices].max():.4f}")
    print(f"Resultados salvos em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
