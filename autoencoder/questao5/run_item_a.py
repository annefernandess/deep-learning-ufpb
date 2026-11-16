"""
Orquestração do item (a): treina 3 autoencoders (latent_dim=100,50,30),
avalia cada um nas mesmas 10 imagens fixas (1 por classe) e gera a figura
comparativa + tabela resumo de erro médio.
"""

import os

from data import NOMES_CLASSES, obter_datasets, obter_imagens_fixas, obter_train_loader
from evaluate import avaliar_experimento
from visualize import plotar_comparacao_latentes

NUM_EPOCAS = 20
LATENT_DIMS = [100, 50, 30]
DIRETORIO_CHECKPOINTS = "checkpoints"
DIRETORIO_OUTPUTS = "outputs"
SEED = 42


def main():
    os.makedirs(DIRETORIO_CHECKPOINTS, exist_ok=True)
    os.makedirs(DIRETORIO_OUTPUTS, exist_ok=True)

    train_dataset, test_dataset = obter_datasets()
    train_loader = obter_train_loader(train_dataset)

    # Mesmas 10 imagens (1 por classe) que serão reutilizadas no item (b).
    imagens_fixas, labels_fixas, _indices = obter_imagens_fixas(test_dataset, seed=SEED)

    resultados = {}
    for latent_dim in LATENT_DIMS:
        print(f"\n=== Item (a) — latent_dim={latent_dim} ===")
        resultados[latent_dim] = avaliar_experimento(
            latent_dim=latent_dim,
            train_loader=train_loader,
            imagens_fixas=imagens_fixas,
            num_epocas=NUM_EPOCAS,
            diretorio_checkpoints=DIRETORIO_CHECKPOINTS,
            seed=SEED,
        )

    plotar_comparacao_latentes(
        imagens_originais=imagens_fixas,
        resultados=resultados,
        labels_classes=labels_fixas,
        nomes_classes=NOMES_CLASSES,
        caminho_saida=os.path.join(DIRETORIO_OUTPUTS, "comparacao_item_a.png"),
    )

    print("\n=== Resumo: erro médio de reconstrução (MSE) por latent_dim ===")
    for latent_dim in LATENT_DIMS:
        erro_medio = resultados[latent_dim]["erro_por_imagem"].mean().item()
        print(f"latent_dim={latent_dim:>4}: erro médio = {erro_medio:.6f}")


if __name__ == "__main__":
    main()
