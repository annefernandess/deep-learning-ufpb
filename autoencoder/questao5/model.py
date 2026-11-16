"""
Arquitetura do autoencoder convolucional.

Única responsabilidade deste módulo: a definição da rede (encoder + decoder).
Nada de treino ou carregamento de dados deve viver aqui.
"""

import torch.nn as nn


class Autoencoder(nn.Module):
    """Autoencoder convolucional com espaço latente de tamanho configurável.

    `latent_dim` é o único parâmetro que varia entre as 3 versões pedidas no
    item (a) (100, 50, 30) — a arquitetura (número de camadas e canais) é
    idêntica entre elas; apenas o tamanho do "gargalo" muda.
    """

    def __init__(self, latent_dim: int):
        super().__init__()

        # Encoder: comprime a imagem (1,28,28) espacialmente enquanto aumenta
        # o número de canais, até achatar tudo em um vetor latente.
        # Fórmula de cada Conv2d (stride=2, padding=1, kernel=3):
        #   saida = floor((entrada + 2*padding - kernel) / stride) + 1
        # 28 -> floor((28+2-3)/2)+1 = floor(13.5)+1 = 14
        # 14 -> floor((14+2-3)/2)+1 = floor(6.5)+1  = 7
        #  7 -> floor((7+2-3)/2)+1  = floor(3.0)+1  = 4
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),   # (1,28,28)  -> (16,14,14)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (16,14,14) -> (32,7,7)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (32,7,7)   -> (64,4,4)
            nn.ReLU(),
            nn.Flatten(),                                           # (64,4,4)   -> (1024,)
            nn.Linear(64 * 4 * 4, latent_dim),                       # (1024,)    -> (latent_dim,)
            # Sem ativação aqui: o vetor latente fica livre para assumir
            # qualquer valor real, sem restrição de sinal/faixa.
        )

        # Decoder: espelha exatamente o encoder, expandindo de volta para (1,28,28).
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 4 * 4),                       # (latent_dim,) -> (1024,)
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(64, 4, 4)),        # (1024,) -> (64,4,4)
            # ConvTranspose2d: saida = (entrada-1)*stride - 2*padding + kernel + output_padding
            # output_padding compensa o arredondamento (floor) que ocorreu na
            # conv correspondente do encoder:
            #   camada 1 (4->7): encoder 7->4 foi divisão exata (6/2=3) -> output_padding=0
            #       saida = (4-1)*2 - 2 + 3 + 0 = 7
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),                                               # (64,4,4) -> (32,7,7)
            #   camada 2 (7->14): encoder 14->7 arredondou (13/2=6.5->6) -> output_padding=1
            #       saida = (7-1)*2 - 2 + 3 + 1 = 14
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),                                               # (32,7,7) -> (16,14,14)
            #   camada 3 (14->28): encoder 28->14 arredondou (27/2=13.5->13) -> output_padding=1
            #       saida = (14-1)*2 - 2 + 3 + 1 = 28
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # restringe a saída a [0,1], compatível com a entrada normalizada
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_reconstruido = self.decode(z)
        return x_reconstruido
