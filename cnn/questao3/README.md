# Questão 3 — MLP × CNN no CIFAR-10

Dois classificadores sobre o CIFAR-10: (a) MLP e (b) CNN.

## Arquiteturas

- **MLP**: `3072 → 512 → 256 → 10`, com ReLU e Dropout (0,3) entre as camadas ocultas.
- **CNN**: três blocos `Conv2D(3×3) + BatchNorm + ReLU + MaxPool`, com 32, 64 e 128 canais; pooling adaptativo e classificador `128 → 128 → 10` com ReLU e Dropout (0,3).

Ambos usam Cross-Entropy Loss e Adam (`lr=0,001`). Dos 50.000 exemplos oficiais de
treino, 10% são separados estratificadamente para validação; o conjunto de teste
oficial permanece separado e é usado uma única vez para a avaliação final.

## Carregamento e preparação dos dados

`data.py` baixa e extrai o CIFAR-10 em `cnn/data/` na primeira execução. `ToTensor`
converte pixels de `[0, 255]` para `float32` em `[0, 1]` e mantém a ordem
`(canais, altura, largura)` para a CNN. A visão da MLP achata cada imagem para
3072 elementos.

```python
from cnn.questao3.data import load_cifar10

train_cnn, test_cnn, train_mlp, test_mlp = load_cifar10()

image_cnn, label = train_cnn[0]  # (3, 32, 32)
image_mlp, _ = train_mlp[0]      # (3072,)
```

## Treinamento e saídas

Na raiz do repositório, com o ambiente virtual ativado:

```bash
python -m cnn.questao3.train
```

Por padrão, o MLP treina por 5 épocas e a CNN por 8. Os valores podem ser
alterados com `--mlp-epochs`, `--cnn-epochs` e `--batch-size`. O melhor checkpoint
da CNN, selecionado pela menor loss de validação, fica em
`cnn/questao3/checkpoints/cifar10_cnn.pt` para reutilização na Questão 4.

O diretório `cnn/questao3/results/` recebe as curvas de loss
(`loss_curves.png`), as matrizes de confusão (`mlp_confusion_matrix.png` e
`cnn_confusion_matrix.png`) e `metrics.json` com losses, acurácias de teste,
matrizes e histórico por época.

## Resultados obtidos

Execução em CPU, com seed 42 e batch size 256:

| Modelo | Épocas | Melhor época (validação) | Loss de teste | Acurácia de teste |
| --- | ---: | ---: | ---: | ---: |
| MLP | 5 | 5 | 1,5795 | 43,79% |
| CNN | 8 | 5 | 1,0910 | 60,89% |

O checkpoint salvo corresponde à melhor época de validação da CNN (época 5).
