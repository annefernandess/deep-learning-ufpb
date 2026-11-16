# Questão 4 — Interpretabilidade da CNN (CIFAR-10)

Usa a CNN da Questão 3 e o checkpoint `cnn/questao3/checkpoints/cifar10_cnn.pt`.

Execute da raiz do repositório, com o ambiente virtual ativado:

```bash
python -m cnn.questao4.interpret
```

O script coloca os arquivos em `cnn/questao4/results/`:

- `first_layer_filters.png`: visualização dos 32 filtros RGB da primeira convolução, normalizados individualmente para exibição.
- `test_feature_maps.png`: cinco imagens de teste corretamente classificadas; em cada linha, a imagem original e overlays dos quatro canais com maior ativação média após a segunda convolução (`features[6]`, 64 canais, resolução 16×16).
- `classification_examples.png`: cinco acertos de alta confiança (um por classe sempre que disponível) e os cinco erros com maior confiança; cada título informa classe verdadeira, classe predita e probabilidade softmax.
- `selected_examples.csv`: índices e metadados das amostras e canais visualizados.

Os mapas são ativações da camada convolucional, não uma explicação causal da decisão. Regiões mais intensas indicam onde os canais selecionados responderam mais fortemente; a resolução 16×16 é ampliada para sobreposição à imagem de 32×32.

## Análise qualitativa

Os 32 filtros RGB exibem combinações variadas de cores locais. Como cada kernel tem apenas 3×3 pixels e é normalizado individualmente para visualização, não é possível afirmar que todos sejam detectores de borda específicos; alguns mostram contrastes direcionais ou entre canais de cor, mas os padrões são abstratos.

Nos mapas de ativação, a amostra `airplane` (índice 235) destaca a silhueta e a faixa das asas; `automobile` (7876), o teto e partes contrastantes da carroceria; `bird` (3908), o contorno do corpo; `cat` (9228), regiões do corpo e da cabeça; e `deer` (2655), áreas do corpo em contraste com o fundo. Também há respostas fortes no fundo e nas bordas, mostrando que os canais visualizados não são exclusivos da classe.

Entre os erros de maior confiança, `cat → frog` (98,89%), `horse → truck` (98,45%), `cat → frog` (98,10%), `airplane → truck` (97,27%) e `deer → frog` (97,23%). Esses casos ilustram que a confiança softmax não garante acerto; baixa resolução, contexto e semelhanças visuais podem contribuir para confusões. Os mapas são respostas de canais selecionados, não uma atribuição causal da decisão da rede.
