"""
Orquestração do item (b): treina um autoencoder denoising (mesma
arquitetura-base do item a, latent_dim=100) e avalia nas mesmas 10 imagens
fixas usadas no item (a), comparando limpa / ruidosa / reconstruída.
"""

import os

from data import NOMES_CLASSES, adicionar_ruido, obter_datasets, obter_imagens_fixas, obter_train_loader
from evaluate import avaliar_denoising
from visualize import plotar_denoising

NUM_EPOCAS = 20
LATENT_DIM = 100  # mesma arquitetura-base do item (a); escolhido o maior latent_dim testado
DIRETORIO_CHECKPOINTS = "checkpoints"
DIRETORIO_OUTPUTS = "outputs"
SEED = 42

# Intensidade do ruído Gaussiano: definida uma única vez aqui e reutilizada
# tanto no treino (usar_ruido=True, std_ruido=STD_RUIDO) quanto na avaliação
# (adicionar_ruido(imagens_fixas, STD_RUIDO)) — nunca repetida como literal.
STD_RUIDO = 0.1


def main():
    os.makedirs(DIRETORIO_CHECKPOINTS, exist_ok=True)
    os.makedirs(DIRETORIO_OUTPUTS, exist_ok=True)

    train_dataset, test_dataset = obter_datasets()
    train_loader = obter_train_loader(train_dataset)

    # Mesma seed do item (a) (seed=42, mesma função obter_imagens_fixas):
    # garante que sejam exatamente as mesmas 10 imagens.
    imagens_fixas, labels_fixas, _indices = obter_imagens_fixas(test_dataset, seed=SEED)
    imagens_ruidosas = adicionar_ruido(imagens_fixas, STD_RUIDO)

    caminho_checkpoint = os.path.join(DIRETORIO_CHECKPOINTS, "autoencoder_denoising.pt")

    print(f"\n=== Item (b) — denoising (latent_dim={LATENT_DIM}, std_ruido={STD_RUIDO}) ===")
    resultado = avaliar_denoising(
        latent_dim=LATENT_DIM,
        train_loader=train_loader,
        imagens_fixas=imagens_fixas,
        imagens_ruidosas=imagens_ruidosas,
        num_epocas=NUM_EPOCAS,
        caminho_checkpoint=caminho_checkpoint,
        std_ruido=STD_RUIDO,
        seed=SEED,
    )

    plotar_denoising(
        imagens_originais=imagens_fixas,
        imagens_ruidosas=imagens_ruidosas,
        reconstrucoes=resultado["reconstrucoes"],
        erro_por_imagem=resultado["erro_por_imagem"],
        labels_classes=labels_fixas,
        nomes_classes=NOMES_CLASSES,
        caminho_saida=os.path.join(DIRETORIO_OUTPUTS, "comparacao_item_b.png"),
    )

    erro_medio = resultado["erro_por_imagem"].mean().item()
    print(f"\nErro médio de reconstrução (denoising): {erro_medio:.6f}")


if __name__ == "__main__":
    main()
