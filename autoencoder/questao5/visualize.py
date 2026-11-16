"""
Plotagem dos resultados (matplotlib).

Única responsabilidade deste módulo: gerar as figuras comparativas.
Nenhuma lógica de modelo/treino deve viver aqui.
"""

import matplotlib.pyplot as plt


def _para_numpy(imagem):
    """Converte um tensor (1,28,28) em um array numpy (28,28) pronto para imshow."""
    return imagem.squeeze().detach().cpu().numpy()


def _remover_ticks(eixo):
    """Remove os ticks/moldura do subplot sem usar axis('off'), que também
    esconderia o set_ylabel usado para rotular cada linha da grade."""
    eixo.set_xticks([])
    eixo.set_yticks([])
    for spine in eixo.spines.values():
        spine.set_visible(False)


def plotar_comparacao_latentes(imagens_originais, resultados, labels_classes, nomes_classes, caminho_saida):
    """Grade comparativa do item (a): 1 linha de originais + 1 linha por
    latent_dim testado, 10 colunas (uma por imagem/categoria).

    `resultados` é um dicionário {latent_dim: {reconstrucoes, erro_por_imagem, ...}}.
    """
    latent_dims = list(resultados.keys())
    num_linhas = 1 + len(latent_dims)
    num_colunas = imagens_originais.shape[0]

    fig, eixos = plt.subplots(num_linhas, num_colunas, figsize=(num_colunas * 1.5, num_linhas * 1.6))

    # Linha 0: imagens originais, rotuladas com o nome da categoria.
    for coluna in range(num_colunas):
        eixo = eixos[0, coluna]
        eixo.imshow(_para_numpy(imagens_originais[coluna]), cmap="gray")
        eixo.set_title(nomes_classes[labels_classes[coluna]], fontsize=8)
        _remover_ticks(eixo)
    eixos[0, 0].set_ylabel("Original", fontsize=9)

    # Uma linha por latent_dim, com a reconstrução e o erro MSE daquela imagem.
    for linha, latent_dim in enumerate(latent_dims, start=1):
        reconstrucoes = resultados[latent_dim]["reconstrucoes"]
        erro_por_imagem = resultados[latent_dim]["erro_por_imagem"]
        for coluna in range(num_colunas):
            eixo = eixos[linha, coluna]
            eixo.imshow(_para_numpy(reconstrucoes[coluna]), cmap="gray")
            eixo.set_title(f"MSE={erro_por_imagem[coluna].item():.4f}", fontsize=7)
            _remover_ticks(eixo)
        eixos[linha, 0].set_ylabel(f"latent_dim={latent_dim}", fontsize=9)

    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print(f"Figura salva em {caminho_saida}")


def plotar_denoising(imagens_originais, imagens_ruidosas, reconstrucoes, erro_por_imagem, labels_classes, nomes_classes, caminho_saida):
    """Grade comparativa do item (b): 3 linhas (original limpa, ruidosa,
    reconstrução), 10 colunas (uma por imagem/categoria)."""
    num_colunas = imagens_originais.shape[0]
    fig, eixos = plt.subplots(3, num_colunas, figsize=(num_colunas * 1.5, 3 * 1.6))

    linhas = [
        ("Original", imagens_originais, None),
        ("Ruidosa", imagens_ruidosas, None),
        ("Reconstruída", reconstrucoes, erro_por_imagem),
    ]

    for linha, (titulo, imagens, erros) in enumerate(linhas):
        for coluna in range(num_colunas):
            eixo = eixos[linha, coluna]
            eixo.imshow(_para_numpy(imagens[coluna]), cmap="gray")
            if linha == 0:
                eixo.set_title(nomes_classes[labels_classes[coluna]], fontsize=8)
            elif erros is not None:
                eixo.set_title(f"MSE={erros[coluna].item():.4f}", fontsize=7)
            _remover_ticks(eixo)
        eixos[linha, 0].set_ylabel(titulo, fontsize=9)

    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print(f"Figura salva em {caminho_saida}")
