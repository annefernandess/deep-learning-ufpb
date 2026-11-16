"""
Carregamento e preparação dos dados do Fashion-MNIST.

Única responsabilidade deste módulo: dataset, transforms, DataLoaders,
seleção das 10 imagens fixas (uma por classe) e geração de ruído sintético.
Nenhuma lógica de modelo ou de treino deve viver aqui.
"""

import random

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Nome de cada classe do Fashion-MNIST, na ordem do índice numérico usado pelo dataset.
# Usado só para rotular os plots (visualize.py) com nomes legíveis em vez de números.
NOMES_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Pasta onde o torchvision baixa/cacheia os arquivos brutos do Fashion-MNIST.
DIRETORIO_DADOS = "./dados"


def obter_datasets():
    """Cria os datasets de treino e teste do Fashion-MNIST.

    ToTensor() converte a imagem PIL (pixels 0-255) em um tensor float com
    valores em [0, 1] — essa é a única normalização aplicada. Pixels em [0,1]
    combinam com a ativação Sigmoid na saída do decoder (ver model.py).
    """
    transform = transforms.ToTensor()
    train_dataset = datasets.FashionMNIST(
        root=DIRETORIO_DADOS, train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root=DIRETORIO_DADOS, train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset


def obter_train_loader(train_dataset, batch_size=128):
    """DataLoader de treino. shuffle=True evita que os batches sejam blocos
    homogêneos por classe e evita que o modelo veja sempre a mesma ordem."""
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def obter_imagens_fixas(test_dataset, seed=42):
    """Seleciona 1 imagem por classe (10 classes) do conjunto de teste,
    sorteada aleatoriamente mas de forma reprodutível (seed fixa).

    Importante: usa o módulo `random` do Python, não `torch.manual_seed`,
    para que esta escolha fique desacoplada de qualquer aleatoriedade do
    PyTorch (inicialização de pesos, shuffle do DataLoader) usada em outras
    partes do pipeline (ver train.py). Isso garante que a mesma seed sempre
    produza as mesmas 10 imagens, independente do que acontece no treino.

    Retorna:
        imagens: tensor (10, 1, 28, 28), uma imagem por classe.
        labels: tensor (10,) com a classe de cada imagem (0 a 9).
        indices: lista com os índices originais em test_dataset, na mesma ordem.
    """
    random.seed(seed)

    # Agrupa os índices do dataset de teste por classe (0 a 9).
    targets = test_dataset.targets
    indices_por_classe = {classe: [] for classe in range(10)}
    for indice, classe in enumerate(targets.tolist()):
        indices_por_classe[classe].append(indice)

    indices_escolhidos = []
    for classe in range(10):
        indice_sorteado = random.choice(indices_por_classe[classe])
        indices_escolhidos.append(indice_sorteado)

    imagens = torch.stack([test_dataset[i][0] for i in indices_escolhidos])
    labels = torch.tensor([test_dataset[i][1] for i in indices_escolhidos])

    return imagens, labels, indices_escolhidos


def adicionar_ruido(imagens, std):
    """Adiciona ruído Gaussiano às imagens e satura o resultado em [0, 1].

    `std` não tem valor default: deve ser passado explicitamente em toda
    chamada, para forçar quem chama (treino e avaliação do item b) a usar a
    mesma constante STD_RUIDO definida uma única vez em run_item_b.py.
    """
    imagens_ruidosas = imagens + std * torch.randn_like(imagens)
    # Satura de volta para [0,1]: sem isso os valores saem da faixa esperada
    # pela rede (mesma faixa usada na normalização de entrada, ver ToTensor()).
    imagens_ruidosas = torch.clamp(imagens_ruidosas, 0.0, 1.0)
    return imagens_ruidosas
