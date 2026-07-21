# Projeto 1 — Classificação MNIST

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos (0-9)**, e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

## 🎯 Conjunto de Dados

Dataset **MNIST**, disponível diretamente via `tf.keras.datasets.mnist` (não é necessário download manual).

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset MNIST via TensorFlow
- **Split explícito treino/validação** (ex: `validation_split` ou um split manual)
- Construção de uma CNN com:
  - **3 a 4 blocos convolucionais** (`Conv2D` + `BatchNormalization` + `MaxPooling2D`)
  - Camada de `Dropout` antes da saída, para regularização
- Treinamento com **early stopping** baseado na perda de validação (`EarlyStopping`)
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de Edge AI.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/1-classificacao-mnist/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 28x28, 1 canal (grayscale), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 15, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo: Hiago de Oliveira Lima**

### 1️⃣ Resumo da Arquitetura do Modelo

A CNN possui 3 blocos convolucionais, cada um formado por uma camada Conv2D (3x3, ativação ReLU), seguida de BatchNormalization e MaxPooling2D (2x2). O BatchNorm estabiliza o treino e acelera a convergência; o MaxPooling reduz o custo computacional, o que é relevante já que o treino é restrito a CPU.

Optou-se por 3 blocos, e não 4, porque a restrição do desafio pede uma arquitetura simples, e porque, os 3 blocos já reduzem a imagem a 3x3, o que faria um quarto bloco não ser tão viável nesse ponto.

Depois dos blocos convolucionais, a rede aplica um Flatten, seguido de uma camada densa de 128 neurônios com ReLU e um Dropout de 0.4 (que desliga 40% dos neurônios durante o treino, reduzindo overfitting), até chegar na camada de saída, uma Dense de 10 neurônios com softmax.

A validação foi feita com um split manual e embaralhado (85% treino, 15% validação), usando uma seed fixa (np.random.default_rng) para garantir reprodutibilidade e evitar o viés de um corte sequencial sem embaralhamento.

O EarlyStopping monitora a val_loss, com patience = 3 (em vez do padrão, que é 0, e pararia o treino já na primeira flutuação) e restore_best_weights = True (que mantém os pesos da melhor época, não os da última). O treino parou na época 7 de um teto de 15, confirmando que o mecanismo funcionou como esperado. Vale registrar como limitação da técnica que restore_best_weights=True só garante restaurar os melhores pesos se o EarlyStopping for de fato acionado antes do teto de épocas, o que ocorreu normalmente neste treino.

### 2️⃣ Bibliotecas Utilizadas

TensorFlow **2.15.0**

Numpy **1.26.4**

### 3️⃣ Técnica de Otimização do Modelo

A técnica utilizada foi a Quantização de Faixa Dinâmica, aplicada através do TFLiteConverter com `converter.optimizations = [tf.lite.Optimize.DEFAULT]`.

Essa técnica converte os pesos do modelo de ponto flutuante de 32 bits para inteiros de 8 bits, reduzindo o tamanho do arquivo em até 4 vezes. As ativações (cálculos realizados durante a inferência) permanecem em float32, sendo convertidas dinamicamente durante a execução.

Por afetar apenas os pesos, essa técnica reduz o tamanho do modelo com perda de acurácia geralmente pequena, sem exigir um dataset representativo adicional.

### 4️⃣ Resultados Obtidos

O modelo atingiu 98,89% de acurácia de validação (loss de validação: 0,0425), com o treino interrompido pelo EarlyStopping na época 7 de um teto de 15. Além disso, o restore_best_weights nesse treino restaurou os pesos da época 5, batendo exatamente com o valor final impresso.

Tamanho dos artefatos gerados:

_model.h5:_ **2913,79 KB (~2,9 MB)**

_model.tflite:_ **247,73 KB**

Redução de **91,5%** após a quantização de faixa dinâmica.

### 5️⃣ Comentários Adicionais (Opcional)

A maior dificuldade do desafio não esteve na modelagem em si, mas em identificar um problema de compatibilidade entre versões: o model.h5, salvo inicialmente com TensorFlow 2.21.0 (que usa Keras 3 por padrão), não conseguia ser recarregado nem pelo optimize_model.py, nem pelo script de validação automática do GitHub Actions, apresentando o erro `GlorotUniform.__init__()` got an unexpected keyword argument 'input_axes'.

Diagnosticar a causa levou algumas tentativas: inicialmente suspeitou-se de descompasso de versão entre o ambiente local e o CI, mas ambos mostraram estar na mesma versão (2.21.0), descartando essa hipótese.

Um relato semelhante — envolvendo outro parâmetro (batch_shape) mas o mesmo padrão de erro ao carregar modelos Keras 3 em .h5 foi encontrado em uma discussão da comunidade ST sobre deploy em Edge AI (https://community.st.com/t5/edge-ai/unrecognized-keyword-arguments-batch-shape-with-loading-keras/td-p/650324), reforçando que se tratava de um bug conhecido de auto-inconsistência do Keras 3 no formato .h5 legado. A rotina de escrita inclui parâmetros novos do inicializador que a rotina de leitura desse mesmo formato não repassa corretamente ao reconstruir a camada.

A solução foi fixar tensorflow == 2.15.0 no requirements.txt que a ultima versão que usa o Keras 2, evitando o Keras 3, cujo formato .h5 é maduro e não apresenta esse problema.

Antes de chegar nessa solução, também foi testado forçar os.environ["TF_USE_LEGACY_KERAS"] = "1" sem alterar o requirements.txt, o que não funcionou por depender do pacote tf_keras, não instalado no ambiente, confirmando que a correção precisava necessariamente passar pelas dependências declaradas, não só pelo código dos scripts.

Como aprendizado, esse episódio deixou claro que "mesma versão instalada nos dois ambientes" não é garantia suficiente de reprodutibilidade, o próprio formato de arquivo escolhido (.h5 legado vs. .keras nativo) pode ter bugs internos independentes de qualquer descompasso entre máquinas.

### 6️⃣ Exemplo de Inferência

Rodando inferencia em 10 amostras usando model.tflite:

Amostra 1: Predita = 7 , Real = 7 [OK]

Amostra 2: Predita = 2 , Real = 2 [OK]

Amostra 3: Predita = 1 , Real = 1 [OK]

Amostra 4: Predita = 0 , Real = 0 [OK]

Amostra 5: Predita = 4 , Real = 4 [OK]

Amostra 6: Predita = 1 , Real = 1 [OK]

Amostra 7: Predita = 4 , Real = 4 [OK]

Amostra 8: Predita = 9 , Real = 9 [OK]

Amostra 9: Predita = 5 , Real = 5 [OK]

Amostra 10: Predita = 9 , Real = 9 [OK]

Acurácia: 100.00% (10/10)

**Rodando com 500 amostras temos**

**...**

Amostra 446: Predita = 0 , Real = 6 [ERRO]

Amostra 447: Predita = 6 , Real = 6 [OK]

Amostra 448: Predita = 4 , Real = 4 [OK]

Amostra 449: Predita = 9 , Real = 9 [OK]

Amostra 450: Predita = 5 , Real = 3 [ERRO]

Amostra 451: Predita = 3 , Real = 3 [OK]

Amostra 452: Predita = 3 , Real = 3 [OK]

Amostra 453: Predita = 2 , Real = 2 [OK]

Amostra 454: Predita = 3 , Real = 3 [OK]

Amostra 455: Predita = 9 , Real = 9 [OK]

Amostra 456: Predita = 1 , Real = 1 [OK]

Amostra 457: Predita = 2 , Real = 2 [OK]

Amostra 458: Predita = 6 , Real = 6 [OK]

Amostra 459: Predita = 8 , Real = 8 [OK]

Amostra 460: Predita = 0 , Real = 0 [OK]

Amostra 461: Predita = 5 , Real = 5 [OK]

Amostra 462: Predita = 6 , Real = 6 [OK]

Amostra 463: Predita = 6 , Real = 6 [OK]

Amostra 464: Predita = 6 , Real = 6 [OK]

Amostra 465: Predita = 3 , Real = 3 [OK]

Amostra 466: Predita = 8 , Real = 8 [OK]

Amostra 467: Predita = 8 , Real = 8 [OK]

Amostra 468: Predita = 2 , Real = 2 [OK]

Amostra 469: Predita = 7 , Real = 7 [OK]

Amostra 470: Predita = 5 , Real = 5 [OK]

Amostra 471: Predita = 8 , Real = 8 [OK]

Amostra 472: Predita = 9 , Real = 9 [OK]

Amostra 473: Predita = 6 , Real = 6 [OK]

Amostra 474: Predita = 1 , Real = 1 [OK]

Amostra 475: Predita = 8 , Real = 8 [OK]

Amostra 476: Predita = 4 , Real = 4 [OK]

Amostra 477: Predita = 1 , Real = 1 [OK]

Amostra 478: Predita = 2 , Real = 2 [OK]

Amostra 479: Predita = 5 , Real = 5 [OK]

Amostra 480: Predita = 9 , Real = 9 [OK]

Amostra 481: Predita = 1 , Real = 1 [OK]

Amostra 482: Predita = 9 , Real = 9 [OK]

Amostra 483: Predita = 7 , Real = 7 [OK]

Amostra 484: Predita = 5 , Real = 5 [OK]

Amostra 485: Predita = 4 , Real = 4 [OK]

Amostra 486: Predita = 0 , Real = 0 [OK]

Amostra 487: Predita = 8 , Real = 8 [OK]

Amostra 488: Predita = 9 , Real = 9 [OK]

Amostra 489: Predita = 9 , Real = 9 [OK]

Amostra 490: Predita = 1 , Real = 1 [OK]

Amostra 491: Predita = 0 , Real = 0 [OK]

Amostra 492: Predita = 5 , Real = 5 [OK]

Amostra 493: Predita = 2 , Real = 2 [OK]

Amostra 494: Predita = 3 , Real = 3 [OK]

Amostra 495: Predita = 7 , Real = 7 [OK]

Amostra 496: Predita = 8 , Real = 8 [OK]

Amostra 497: Predita = 9 , Real = 9 [OK]

Amostra 498: Predita = 4 , Real = 4 [OK]

Amostra 499: Predita = 0 , Real = 0 [OK]

Amostra 500: Predita = 6 , Real = 6 [OK]

Com apenas 10 amostras, o acerto de 100% não é muito representativo, em uma amostra maior (500 imagens), a acurácia observada foi de 99,20% (496/500), próxima aos 99,02% medidos na validação durante o treino, confirmando que a quantização não degradou o desempenho do modelo.

Dois erros se destacam: a amostra 450 (3 predito como 5) é consistente com um padrão documentado na literatura, 3 e 5 estão entre os pares de dígitos mais frequentemente confundidos em classificadores treinados no MNIST (https://arxiv.org/pdf/2411.12127).

Já a amostra 446 (6 predito como 0) parece um caso pontual dessa imagem específica, sem um padrão sistemático claro na literatura consultada.

### Referências

https://keras.io/api/layers/

https://keras.io/api/callbacks/early_stopping/

https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_quantization

https://pub.aimind.so/never-use-restore-best-weights-true-with-earlystopping-754ba5f9b0c6

#### Materias do curso

📘 Fundamentos de Inteligência Artificial para Sistemas Embarcados

👁️ Sistemas de Visão Computacional Embarcada

⚙️ Otimização de Modelos em Sistemas Embarcados
