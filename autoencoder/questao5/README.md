## Questão 5

Utilize o dataset Fashion-MNIST, disponível em
https://www.kaggle.com/datasets/zalando-research/fashionmnist,
para realizar as atividades descritas a seguir.

### a)

Implemente e treine um autoencoder capaz de reconstruir as imagens do conjunto de dados.
O modelo pode ser construído utilizando apenas camadas densas (*fully connected*), ou
camadas convolucionais e deconvolucionais.

Treine pelo menos **três versões do modelo** utilizando diferentes tamanhos para o espaço
latente (ex: vetor latente com 100, 50 e 30 componentes).

Após o treinamento, apresente:

- arquitetura utilizada;
- selecione 10 imagens do conjunto de teste e compare os erros de reconstrução e
  qualidade das imagens reconstruídas;
- apresente visualmente as imagens originais e reconstruídas.

### b)

Adicione ruído aleatório às imagens de entrada e treine um novo autoencoder com o objetivo
de remover esse ruído. Após o treinamento, utilize as mesmas 10 imagens selecionadas
anteriormente e apresente as imagens ruidosas e as reconstruções produzidas pelo modelo.