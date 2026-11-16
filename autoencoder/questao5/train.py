"""
Loop de treino genérico, reutilizado pelo item (a) e pelo item (b).

Única responsabilidade deste módulo: o loop de treino. Não sabe nada sobre
qual modelo específico (latent_dim) ou qual item está sendo treinado — isso
é decidido por quem chama (run_item_a.py / run_item_b.py).
"""

from data import adicionar_ruido


def treinar(modelo, data_loader, otimizador, criterio, num_epocas, usar_ruido=False, std_ruido=None):
    """Treina `modelo` por `num_epocas` sobre `data_loader`, minimizando o
    erro de reconstrução (`criterio`, tipicamente MSELoss) entre a saída da
    rede e a imagem original.

    Quando usar_ruido=True (item b, denoising), a rede recebe a imagem com
    ruído como entrada mas o alvo continua sendo a imagem limpa — assim ela
    aprende a remover o ruído, não só a copiar a entrada. `std_ruido` é
    obrigatório nesse caso (sem default), para forçar quem chama a usar
    explicitamente a mesma constante STD_RUIDO usada depois na avaliação.
    """
    modelo.train()
    historico_perda = []

    for epoca in range(num_epocas):
        perda_acumulada = 0.0
        for imagens, _rotulos in data_loader:  # rótulo de classe é ignorado (autoencoder não usa)
            entrada = imagens
            alvo = imagens
            if usar_ruido:
                entrada = adicionar_ruido(imagens, std=std_ruido)

            reconstrucao = modelo(entrada)
            perda = criterio(reconstrucao, alvo)

            otimizador.zero_grad()
            perda.backward()
            otimizador.step()

            perda_acumulada += perda.item()

        perda_media_epoca = perda_acumulada / len(data_loader)
        historico_perda.append(perda_media_epoca)
        print(f"  Época {epoca + 1}/{num_epocas} - perda média: {perda_media_epoca:.6f}")

    return historico_perda
