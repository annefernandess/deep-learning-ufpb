# deep-learning-ufpb

Resoluções da lista de exercícios de Aprendizado Profundo (2026.2, Prof. Tiago Maritan, UFPB). Enunciado em [docs/DL_Lista_de_Exercicios.pdf](docs/DL_Lista_de_Exercicios.pdf).

## Estrutura

| Pasta                                         | Questões | Tema                                              |
| --------------------------------------------- | -------- | ------------------------------------------------- |
| [mlp/questao1](mlp/questao1/)                 | 1        | MLP: aproximação de função e regressão (Diabetes) |
| [mlp/questao2](mlp/questao2/)                 | 2        | MLP: classificação de qualidade de vinhos         |
| [cnn/questao3](cnn/questao3/)                 | 3        | MLP × CNN no CIFAR-10                             |
| [cnn/questao4](cnn/questao4/)                 | 4        | Interpretabilidade da CNN                         |
| [autoencoder/questao5](autoencoder/questao5/) | 5        | Autoencoders no Fashion-MNIST                     |

Cada pasta de questão tem um README com o checklist do que deve ser entregue.

## Ambiente

Requisitos: Python 3.12 e [requirements.txt](requirements.txt)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Bibliotecas escolhidas:

- **PyTorch + torchvision**: MLP, CNN e autoencoders; datasets CIFAR-10 e Fashion-MNIST com download automático; hooks para extrair ativações e filtros (Questão 4).
- **scikit-learn**: dataset Diabetes, normalização, divisão treino/teste e métricas (MAE, MSE, RMSE, matriz de confusão, precision/recall/f1).
- **imbalanced-learn**: técnicas de balanceamento pedidas na Questão 2.
- **pandas**, **numpy**: análise exploratória e manipulação.
- **matplotlib**: todos os gráficos.
- **ucimlrepo**: download do Wine Quality (UCI), já que as URLs diretas do UCI mudam com frequência.

Observações:

- Em Linux com GPU NVIDIA, o `pip install torch` puxa as dependências CUDA e fica bem maior. Para CPU, use o índice do PyTorch (`--index-url https://download.pytorch.org/whl/cpu`) antes do `requirements.txt`
