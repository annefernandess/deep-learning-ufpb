"""
Métricas e orquestração da avaliação pós-treino.

Única responsabilidade deste módulo: calcular o erro de reconstrução e
organizar os resultados de cada experimento. Nenhuma lógica de plot
(matplotlib) deve viver aqui.
"""

import os

import torch
import torch.nn as nn

from model import Autoencoder
from train import treinar


def calcular_erro_por_imagem(modelo, entrada, alvo=None):
    """Calcula a reconstrução e o erro MSE de cada imagem individualmente.

    No item (a), entrada e alvo são a mesma imagem (alvo=None assume isso
    automaticamente). No item (b), entrada é a imagem ruidosa (o que o
    modelo de fato recebe) e alvo é a imagem limpa (contra o que o erro de
    reconstrução deve ser medido) — os dois precisam ser informados
    explicitamente nesse caso.
    """
    if alvo is None:
        alvo = entrada

    modelo.eval()
    with torch.no_grad():
        reconstrucoes = modelo(entrada)
        criterio = nn.MSELoss(reduction="none")  # não reduz: mantém erro por pixel
        erro_por_pixel = criterio(reconstrucoes, alvo)
        erro_por_imagem = erro_por_pixel.mean(dim=[1, 2, 3])  # média só dentro de cada imagem

    return reconstrucoes, erro_por_imagem


def obter_ou_treinar_modelo(
    latent_dim, caminho_checkpoint, train_loader, num_epocas, seed=42, usar_ruido=False, std_ruido=None
):
    """Para um dado latent_dim: carrega o checkpoint se já existir, ou treina
    do zero e salva. Retorna (modelo, historico_perda) — historico_perda é
    None quando o modelo foi carregado de checkpoint (não há histórico
    salvo junto com os pesos).

    `usar_ruido`/`std_ruido` são passados direto para `treinar` (ver
    train.py) — usados pelo item (b), denoising; irrelevantes no item (a).
    """
    if os.path.exists(caminho_checkpoint):
        print(f"Checkpoint encontrado em {caminho_checkpoint} — carregando pesos (sem re-treinar).")
        modelo = Autoencoder(latent_dim=latent_dim)
        modelo.load_state_dict(torch.load(caminho_checkpoint))
        historico_perda = None
    else:
        print(f"Nenhum checkpoint em {caminho_checkpoint} — treinando latent_dim={latent_dim}.")
        # Mesma seed a cada iteração: garante que a única diferença entre as
        # 3 variantes do item (a) seja latent_dim, não a inicialização dos pesos.
        torch.manual_seed(seed)
        modelo = Autoencoder(latent_dim=latent_dim)
        otimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)
        criterio = nn.MSELoss()
        historico_perda = treinar(
            modelo, train_loader, otimizador, criterio, num_epocas, usar_ruido=usar_ruido, std_ruido=std_ruido
        )
        os.makedirs(os.path.dirname(caminho_checkpoint), exist_ok=True)
        torch.save(modelo.state_dict(), caminho_checkpoint)

    return modelo, historico_perda


def avaliar_experimento(latent_dim, train_loader, imagens_fixas, num_epocas, diretorio_checkpoints, seed=42):
    """Treina (ou recarrega) o autoencoder para um latent_dim, avalia nas 10
    imagens fixas e devolve um dicionário com todos os resultados desse
    experimento — isola a única variável entre execuções (latent_dim) sem
    duplicar a lógica de treino/avaliação. Usado pelo item (a)."""
    caminho_checkpoint = os.path.join(diretorio_checkpoints, f"autoencoder_latent{latent_dim}.pt")
    modelo, historico_perda = obter_ou_treinar_modelo(
        latent_dim, caminho_checkpoint, train_loader, num_epocas, seed=seed
    )

    reconstrucoes, erro_por_imagem = calcular_erro_por_imagem(modelo, imagens_fixas)

    return {
        "latent_dim": latent_dim,
        "modelo": modelo,
        "historico_perda": historico_perda,
        "reconstrucoes": reconstrucoes,
        "erro_por_imagem": erro_por_imagem,
    }


def avaliar_denoising(
    latent_dim, train_loader, imagens_fixas, imagens_ruidosas, num_epocas, caminho_checkpoint, std_ruido, seed=42
):
    """Treina (ou recarrega) o autoencoder denoising e avalia nas 10 imagens
    fixas: entrada é a versão ruidosa, alvo é a versão limpa original. Usado
    pelo item (b) — mesma arquitetura/loop do item (a), única diferença é
    usar_ruido=True no treino e entrada != alvo na avaliação."""
    modelo, historico_perda = obter_ou_treinar_modelo(
        latent_dim,
        caminho_checkpoint,
        train_loader,
        num_epocas,
        seed=seed,
        usar_ruido=True,
        std_ruido=std_ruido,
    )

    reconstrucoes, erro_por_imagem = calcular_erro_por_imagem(modelo, imagens_ruidosas, alvo=imagens_fixas)

    return {
        "latent_dim": latent_dim,
        "modelo": modelo,
        "historico_perda": historico_perda,
        "reconstrucoes": reconstrucoes,
        "erro_por_imagem": erro_por_imagem,
    }
