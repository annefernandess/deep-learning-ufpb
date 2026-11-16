# Questão 3 — MLP × CNN no CIFAR-10

Dois classificadores sobre o CIFAR-10: (a) MLP e (b) CNN.

Entregar, para ambos: arquitetura, curvas de treino e validação, matriz de confusão, acurácia final no teste.
Salvar o checkpoint da CNN treinada, pois a Questão 4 reutiliza esse modelo.

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
