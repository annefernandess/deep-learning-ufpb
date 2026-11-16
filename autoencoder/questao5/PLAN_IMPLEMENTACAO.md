# Plano de Implementação — Questão 5 (Autoencoder, Fashion-MNIST)

> Este documento consolida as decisões teóricas e de design discutidas e aprovadas antes da
> implementação. Serve como especificação para quem (pessoa ou LLM) for escrever o código.
> Todo o código deve conter **comentários explicando o que cada parte faz** (nível didático,
> já que este é um exercício de aprendizado) — não apenas comentários de "por quê" não-óbvio.

## 0. Contexto e requisitos do enunciado

Dataset: Fashion-MNIST (28x28, grayscale, 10 classes).

- **Item (a)**: autoencoder convolucional, treinado em 3 variantes com `latent_dim` em
  `{100, 50, 30}`. Para cada variante, apresentar: arquitetura usada, comparação do erro de
  reconstrução e qualidade visual em 10 imagens de teste, visualização original vs. reconstruída.
- **Item (b)**: autoencoder denoising — mesma arquitetura-base, treinado para remover ruído
  adicionado artificialmente à entrada. Avaliar nas **mesmas 10 imagens** usadas no item (a),
  mostrando versão ruidosa e reconstrução.

Decisão importante: as 10 imagens de teste usadas na avaliação devem ser **1 imagem por classe
(10 classes)**, selecionadas **aleatoriamente mas com seed fixa**, calculadas em runtime a
partir de `dataset.targets` — nunca hardcodadas como lista literal de índices no código.

## 1. Ambiente e bibliotecas

Já existentes em `requirements.txt` da raiz do projeto: `torch`, `torchvision`, `matplotlib`,
`numpy`. Usar `torchvision.datasets.FashionMNIST` (não o CSV do Kaggle) — já vem com
`train`/`test` separados e labels acessíveis via `dataset.targets`.

## 2. Estrutura de pastas e arquivos

```
autoencoder/questao5/
├── README.md                  (já existe — enunciado, não alterar)
├── PLAN_IMPLEMENTACAO.md       (este arquivo)
├── data.py                     # dataset, transforms, DataLoaders, geração de ruído, seleção das 10 imagens fixas
├── model.py                    # classe Autoencoder (encoder + decoder), parametrizada por latent_dim
├── train.py                    # função de treino genérica (reusada pelo item a e b)
├── evaluate.py                 # erro de reconstrução por imagem, execução da avaliação
├── visualize.py                 # funções de plotagem (grades originais vs. reconstruções)
├── run_item_a.py                # script orquestrador do item (a): treina 3 latentes, avalia, plota
├── run_item_b.py                # script orquestrador do item (b): treina denoising, avalia, plota
├── checkpoints/                 # pesos salvos (.pt) por experimento, para evitar re-treino
└── outputs/                     # figuras geradas (.png) pelos scripts de visualização
```

Responsabilidade de cada módulo (não misturar):
- `data.py`: única fonte de verdade sobre como os dados são carregados/normalizados/selecionados.
- `model.py`: só a definição de arquitetura — nada de treino ou dados aqui.
- `train.py`: só o loop de treino — agnóstico a qual modelo ou a qual item (a/b) está treinando.
- `evaluate.py`: só métricas e preparação dos resultados pós-treino.
- `visualize.py`: só matplotlib — nenhuma lógica de modelo/treino aqui.
- `run_item_a.py` / `run_item_b.py`: curtos, só chamam as funções dos outros módulos na ordem certa.

## 3. `data.py`

### 3.1 Transform e normalização
- Converter imagem PIL para tensor (`ToTensor()`), o que já normaliza pixels para `[0, 1]`.
- **Não** normalizar para `[-1, 1]` — decisão ligada diretamente à ativação final do decoder
  (ver seção 4.3). Pixels em `[0,1]` + Sigmoid final é a combinação coerente escolhida.

### 3.2 Datasets e DataLoaders
- `train_dataset = FashionMNIST(root=..., train=True, download=True, transform=...)`
- `test_dataset = FashionMNIST(root=..., train=False, download=True, transform=...)`
- `train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)`
  - `shuffle=True` no treino: evita que os batches sejam blocos homogêneos por classe e evita
    que o modelo veja sempre a mesma sequência de dados a cada época.
- `test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)` — **opcional**: só é
  necessário se você quiser monitorar a perda média no conjunto de teste completo durante o
  treino (prática comum, mas não exigida pelo enunciado). Nenhum dos scripts orquestradores
  (`run_item_a.py`/`run_item_b.py`, seções 8/9) depende dele — as 10 imagens fixas usadas na
  avaliação vêm de `test_dataset` diretamente (ver 3.3), não deste `DataLoader`. Se não for usar
  para monitoramento, pode omitir este `DataLoader` por completo.

### 3.3 Seleção das 10 imagens fixas (uma por classe)
Nomes das 10 classes do Fashion-MNIST (índice -> nome), necessário para os plots (seção 7):
```
0: T-shirt/top   1: Trouser     2: Pullover    3: Dress      4: Coat
5: Sandal        6: Shirt       7: Sneaker     8: Bag        9: Ankle boot
```

Função dedicada, ex: `obter_imagens_fixas(test_dataset, seed=42) -> (imagens, labels, indices)`:
1. Fixar a aleatoriedade usando o módulo `random` do Python (`random.seed(seed)`) — **não**
   usar `torch.manual_seed` para este sorteio específico, para manter a escolha dos índices
   desacoplada de qualquer aleatoriedade do PyTorch (inicialização de pesos, shuffle do
   DataLoader) usada em outras partes do pipeline. `torch.manual_seed` é usado separadamente
   em `train.py` (seção 5.2), para outro propósito.
2. Usar `test_dataset.targets` (tensor com a label de cada amostra) para agrupar índices por
   classe (0 a 9).
3. Para cada uma das 10 classes, sortear aleatoriamente **1 índice** dentro do grupo daquela
   classe.
4. Acessar `test_dataset[idx]` para cada índice sorteado e montar um tensor único
   `(10, 1, 28, 28)` com `torch.stack`.
5. Retornar também os índices e labels usados (para poder exibir o nome da categoria nos plots).
6. Esta função é chamada **uma única vez** (mesma seed) e o resultado (tensor de imagens) é
   reutilizado tanto no item (a) quanto no item (b) — garante "mesmas 10 imagens" exigido pelo
   enunciado, sem jamais escrever os índices como literais no código.

### 3.4 Geração de ruído (para o item b)
Função `adicionar_ruido(imagens, std) -> imagens_ruidosas`:
- `imagens_ruidosas = imagens + std * torch.randn_like(imagens)` (ruído Gaussiano).
- `imagens_ruidosas = torch.clamp(imagens_ruidosas, 0.0, 1.0)` — obrigatório: sem isso os
  valores saem do intervalo `[0,1]` esperado pela rede (mesma faixa da normalização de entrada).
- Esta função deve poder ser aplicada tanto a um batch do `DataLoader` (durante o treino do
  item b) quanto diretamente às 10 imagens fixas (durante a avaliação do item b).
- **`std` não tem valor default na função** — é obrigatório passar o valor explicitamente em
  toda chamada. Isso é intencional: o mesmo valor de `std` **precisa** ser usado no treino
  (`run_item_b.py` passando para `treinar(..., std_ruido=STD_RUIDO)`) e na avaliação
  (`adicionar_ruido(imagens_fixas, STD_RUIDO)`). Definir `STD_RUIDO = 0.3` como constante
  **uma única vez**, no topo de `run_item_b.py`, e reutilizá-la em ambas as chamadas — nunca
  escrever `0.3` como literal em mais de um lugar, para não haver risco de treino e avaliação
  usarem intensidades de ruído diferentes.

## 4. `model.py`

### 4.1 Classe `Autoencoder(nn.Module)`
Construtor recebe `latent_dim: int` como parâmetro — é isso que permite instanciar as 3
variantes (100/50/30) sem duplicar a classe. A arquitetura (número de camadas, canais) é
**idêntica** entre as 3 variantes; só a camada `Linear` final do encoder/inicial do decoder
muda de tamanho.

`latent_dim` define o tamanho do vetor latente (o gargalo) e deve ser usado em **dois** pontos,
lendo sempre o mesmo argumento do construtor — nunca escrito como número fixo em nenhum deles:
- `nn.Linear(1024, latent_dim)` — última camada do encoder (define o tamanho da saída).
- `nn.Linear(latent_dim, 1024)` — primeira camada do decoder (precisa casar com a entrada).
Isso garante que mudar `latent_dim` na instanciação (`Autoencoder(latent_dim=50)`) altere as
duas camadas de forma consistente, sem duplicar/dessincronizar a arquitetura entre variantes.

### 4.2 Encoder
Sequência (comprimindo espacialmente, aumentando canais):
```
Conv2d(1  -> 16, kernel_size=3, stride=2, padding=1)  -> ReLU   # (1,28,28) -> (16,14,14)
Conv2d(16 -> 32, kernel_size=3, stride=2, padding=1)  -> ReLU   # (16,14,14) -> (32,7,7)
Conv2d(32 -> 64, kernel_size=3, stride=2, padding=1)  -> ReLU   # (32,7,7) -> (64,4,4)
Flatten()                                                         # (64,4,4) -> (1024,)
Linear(1024 -> latent_dim)                                       # (1024,) -> (latent_dim,)  [SEM ativação]
```
**Sem ativação após a `Linear` final do encoder**: o vetor latente fica livre para assumir
qualquer valor real (positivo ou negativo), sem restrição de sinal ou faixa — é a escolha
padrão para o espaço latente de um autoencoder (diferente da saída final do decoder, que
precisa de Sigmoid para casar com a faixa `[0,1]` dos pixels, ver 4.3). Adicionar ReLU aqui
restringiria o latente a valores não-negativos sem necessidade, reduzindo a capacidade
representacional sem motivo.

**Importante**: calcular/verificar o tamanho espacial exato resultante de cada `Conv2d` com
`stride=2, padding=1, kernel_size=3` (fórmula: `saida = floor((entrada + 2*padding - kernel)/stride) + 1`).
Confirmar que a sequência 28→14→7→4 está correta antes de fixar os números de `Linear`
(`64 * 4 * 4 = 1024`); se a fórmula resultar em dimensão diferente, ajustar o `in_features` do
`Linear` de acordo — isso **precisa fechar exatamente**, senão o código quebra no forward.

### 4.3 Decoder (espelho do encoder)
```
Linear(latent_dim -> 1024) -> ReLU
Unflatten(dim=1, unflattened_size=(64, 4, 4))                    # (1024,) -> (64,4,4)
ConvTranspose2d(64 -> 32, kernel_size=3, stride=2, padding=1, output_padding=0) -> ReLU  # ->(32,7,7)
ConvTranspose2d(32 -> 16, kernel_size=3, stride=2, padding=1, output_padding=1) -> ReLU  # ->(16,14,14)
ConvTranspose2d(16 -> 1,  kernel_size=3, stride=2, padding=1, output_padding=1) -> Sigmoid # ->(1,28,28)
```
- `output_padding` **não é uniforme** entre as camadas — cada valor foi calculado para
  compensar exatamente o arredondamento (floor) que aconteceu na camada correspondente do
  encoder, usando a fórmula `saida = (entrada-1)*stride - 2*padding + kernel_size + output_padding`:
  - Camada 1 (4→7): a conv. correspondente do encoder (7→4) foi uma divisão exata (`6/2=3`,
    sem arredondamento) → `output_padding=0`.
  - Camada 2 (7→14): a conv. correspondente do encoder (14→7) teve arredondamento
    (`13/2=6.5`→6) → precisa `output_padding=1` para recuperar o pixel perdido no floor.
  - Camada 3 (14→28): mesma situação do encoder (28→14, `27/2=13.5`→13) → `output_padding=1`.
  - **Antes de implementar**, validar essas 3 contas manualmente (ou com um tensor de teste)
    para garantir que a saída final é exatamente `(1, 28, 28)`, não `(1, 27, 27)` ou similar.
- **Sigmoid na última camada**: decisão ligada à normalização de entrada em `[0,1]` (seção 3.1)
  — garante que a saída da rede também fique restrita a `[0,1]`, compatível com o MSE contra
  o alvo normalizado da mesma forma.
- **ReLU após o primeiro `Linear(latent_dim -> 1024)`**: simétrico ao encoder — introduz
  não-linearidade antes do `Unflatten`, igual ao papel da ReLU nas `Conv2d` do encoder (ver
  seção 4.2). Diferente do `Linear` final do **encoder** (que fica sem ativação, pois é o
  próprio vetor latente): este `Linear` do decoder não é o latente, é uma etapa intermediária
  de expansão, então recebe ReLU como as demais camadas intermediárias.

### 4.4 `forward`
```
def forward(self, x):
    z = self.encoder(x)
    x_reconstruido = self.decoder(z)
    return x_reconstruido
```
Opcionalmente expor `encode(x)` e `decode(z)` separados (não obrigatório para o enunciado, mas
útil para debug).

## 5. `train.py`

### 5.1 Função de treino genérica
```
def treinar(modelo, data_loader, otimizador, criterio, num_epocas, usar_ruido=False, std_ruido=None):
    # std_ruido só é usado quando usar_ruido=True; nesse caso é obrigatório (sem default),
    # para forçar quem chama a passar explicitamente a mesma constante usada na avaliação.
    modelo.train()
    historico_perda = []
    for epoca in range(num_epocas):
        perda_acumulada = 0.0
        for imagens, _ in data_loader:          # rótulo de classe é ignorado (autoencoder não usa)
            entrada = imagens
            alvo = imagens
            if usar_ruido:
                entrada = adicionar_ruido(imagens, std=std_ruido)   # denoising: entrada != alvo
            reconstrucao = modelo(entrada)
            perda = criterio(reconstrucao, alvo)

            otimizador.zero_grad()
            perda.backward()
            otimizador.step()

            perda_acumulada += perda.item()
        historico_perda.append(perda_acumulada / len(data_loader))
    return historico_perda
```
- `usar_ruido=False` cobre o item (a); `usar_ruido=True` cobre o item (b) — mesma função,
  nenhuma duplicação de loop.
- `criterio = nn.MSELoss()` — perda de reconstrução (regressão sobre pixels contínuos, não
  classificação; por isso não é `CrossEntropyLoss`).
- `otimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)` (valor de `lr` sugerido, ajustável).
- `num_epocas` sugerido entre 10-30 — observar `historico_perda` para checar convergência.

### 5.2 Reprodutibilidade
Antes de cada chamada de `treinar` (uma por `latent_dim`), fixar `torch.manual_seed(42)` para
que a única variável entre os 3 experimentos do item (a) seja o `latent_dim`, não a
inicialização aleatória dos pesos.

## 6. `evaluate.py`

### 6.1 Erro de reconstrução por imagem (opção simples aprovada)
```
def calcular_erro_por_imagem(modelo, entrada, alvo=None):
    # No item (a), entrada e alvo são a mesma imagem: basta passar "entrada".
    # No item (b), entrada = imagem ruidosa (o que o modelo recebe) e
    # alvo = imagem limpa (contra o que medimos o erro) — os dois precisam ser informados.
    if alvo is None:
        alvo = entrada

    modelo.eval()
    with torch.no_grad():
        reconstrucoes = modelo(entrada)
        criterio = nn.MSELoss(reduction='none')     # não reduz: mantém erro por pixel
        erro_por_pixel = criterio(reconstrucoes, alvo)
        erro_por_imagem = erro_por_pixel.mean(dim=[1, 2, 3])   # média só dentro de cada imagem
    return reconstrucoes, erro_por_imagem
```
- `model.eval()` + `torch.no_grad()` obrigatórios na avaliação (sem necessidade de gradiente
  aqui, e garante comportamento de inferência caso alguma camada dependa do modo).
- **Item (a)**: `calcular_erro_por_imagem(modelo, imagens_fixas)` — `alvo` é assumido igual a
  `entrada` automaticamente.
- **Item (b)**: `calcular_erro_por_imagem(modelo, entrada=imagens_ruidosas, alvo=imagens_fixas)`
  — se os dois fossem o mesmo tensor aqui, o erro medido seria "quão parecida a reconstrução
  ficou do ruído", não o erro de reconstrução contra a imagem original que o enunciado pede.

### 6.2 Orquestração da avaliação por experimento
Função que recebe `latent_dim`, treina (via `train.py`), avalia (via 6.1 nas 10 imagens fixas),
e devolve um dicionário com: modelo treinado, histórico de perda, reconstruções das 10 imagens,
erro por imagem. Chamada em loop sobre `[100, 50, 30]`, guardando resultados em
`resultados[latent_dim] = {...}` — isola a única diferença entre as 3 execuções à variável
`latent_dim`, sem duplicar a lógica de experimento.

### 6.3 Checkpoints
Após treinar cada variante, salvar:
```
torch.save(modelo.state_dict(), f"checkpoints/autoencoder_latent{latent_dim}.pt")
```
E no item (b): `checkpoints/autoencoder_denoising.pt`. Os scripts devem verificar se o arquivo
de checkpoint já existe antes de treinar — se existir, carregar com `modelo.load_state_dict(...)`
em vez de re-treinar (permite re-executar avaliação/plots sem pagar o custo de treino de novo).

**Observação sobre `historico_perda` ao recarregar um checkpoint existente**: quando o modelo é
carregado de um checkpoint em vez de treinado, não existe histórico de perda por época
disponível (ele só é gerado durante o loop de `treinar()`). Nesse caso, `historico_perda` deve
ser definido como `None`, e os scripts de plot/relatório devem tratar esse valor (ex: pular a
curva de perda daquele experimento, ou exibir um aviso), em vez de assumir que ele sempre existe.
Isso é aceitável para este exercício — não é necessário salvar/recarregar o histórico de perda
junto com os pesos, a menos que você queira essa curva disponível mesmo após recarregar.

## 7. `visualize.py`

### 7.1 Grade comparativa — item (a)
Função `plotar_comparacao_latentes(imagens_originais, resultados, labels_classes)`:
- Grade de subplots: 1 linha para originais + 1 linha por `latent_dim` (3 linhas) = 4 linhas,
  10 colunas (uma por imagem/categoria).
- Cada subplot de reconstrução anotado com o erro MSE daquela imagem (título ou texto).
- Usar `imagem.squeeze().detach().cpu().numpy()` antes de `imshow` (remove canal, destacável,
  move para CPU, converte para numpy).
- Salvar em `outputs/comparacao_item_a.png`.

### 7.2 Grade comparativa — item (b)
Função `plotar_denoising(imagens_originais, imagens_ruidosas, reconstrucoes, erros)`:
- 3 linhas: original limpa, ruidosa (entrada real do modelo), reconstrução.
- Mesma lógica de conversão tensor->numpy e anotação de erro.
- Salvar em `outputs/comparacao_item_b.png`.

## 8. `run_item_a.py` (orquestração)
1. Carregar `train_loader`, `test_dataset` (via `data.py`) — não depende de seed.
2. Obter as 10 imagens fixas (uma por classe) via `obter_imagens_fixas` (usa sua própria seed
   interna do módulo `random`, ver 3.3 — independente da seed de treino).
3. Para cada `latent_dim` em `[100, 50, 30]`: checar checkpoint existente → se não existir,
   **fixar `torch.manual_seed(42)` novamente antes de criar e treinar o modelo** (conforme 5.2
   — repetir a mesma seed a cada iteração, não só uma vez no início do script, senão os 3
   treinos não partem de condições iniciais comparáveis) → avaliar nas 10 imagens fixas →
   guardar resultado.
4. Chamar `visualize.plotar_comparacao_latentes(...)`.
5. Imprimir/salvar tabela resumo (ex: erro médio das 10 imagens por `latent_dim`).

## 9. `run_item_b.py` (orquestração)
1. Carregar `train_loader` (mesmos dados do item a).
2. Obter as 10 imagens fixas chamando `obter_imagens_fixas` com o **mesmo `seed` do módulo
   `random`** usado no item (a) (ex: `seed=42`, mesma função de `data.py`) — isso é o que
   garante que sejam as **mesmas 10 imagens**, não a seed de treino.
3. Fixar `torch.manual_seed(42)` antes de criar e treinar o modelo denoising (só uma vez aqui,
   já que o item (b) treina um único modelo — diferente do item (a), que repete isso 3 vezes).
4. Checar checkpoint do autoencoder denoising → se não existir, treinar (`usar_ruido=True`)
   usando a seed do passo 3; se existir, carregar.
5. Gerar versão ruidosa das 10 imagens fixas (`adicionar_ruido`).
6. Avaliar: reconstrução a partir da versão ruidosa, erro contra a versão limpa.
7. Chamar `visualize.plotar_denoising(...)`.

## 10. Checklist de decisões já fechadas (não reabrir sem motivo)
- Arquitetura: convolucional (não dense).
- Normalização: `[0,1]` via `ToTensor()`.
- Ativação final do decoder: Sigmoid (coerente com a normalização acima).
- Função de perda: `MSELoss` (regressão de pixels, não classificação).
- Otimizador: Adam.
- `latent_dim` como parâmetro do construtor do modelo — mesma classe para as 3 variantes.
- Progressão de canais no encoder: 1 → 16 → 32 → 64, com stride 2 em cada camada.
- Simetria decoder = inverso do encoder (`ConvTranspose2d`, mesmos canais ao contrário).
- Erro de reconstrução por imagem: `MSELoss(reduction='none')` + média por imagem (opção simples).
- 10 imagens fixas: 1 por classe, índices sorteados em runtime com seed fixa, nunca hardcoded.
- `model.eval()` + `torch.no_grad()` obrigatórios durante toda avaliação/inferência.
- Checkpoints salvos em `checkpoints/`, carregados se já existirem (evita re-treino redundante).
- Denoising (item b): mesma arquitetura e mesmo loop de treino do item (a); única diferença é
  `entrada = imagem_ruidosa` enquanto `alvo = imagem_limpa`.
- Seleção das 10 imagens fixas usa o módulo `random` do Python (não `torch.manual_seed`),
  desacoplado da aleatoriedade de treino.
- `STD_RUIDO` é uma única constante definida uma vez em `run_item_b.py`, reusada tanto no
  treino quanto na avaliação do item (b) — nunca valores literais repetidos.
- Após a `Linear` final do encoder (vetor latente): sem ativação. Após a `Linear` inicial do
  decoder: com ReLU (é uma etapa intermediária de expansão, não o latente em si).
- `test_loader` (seção 3.2) é opcional — não usado pelos scripts de orquestração, só serviria
  para monitorar perda agregada de teste caso se queira essa análise extra.
- Ao recarregar um checkpoint já treinado, `historico_perda = None` — scripts de plot/relatório
  devem tratar esse caso, não assumir que o histórico sempre existe.
- Código deve conter comentários explicando o que cada trecho faz (nível didático).
