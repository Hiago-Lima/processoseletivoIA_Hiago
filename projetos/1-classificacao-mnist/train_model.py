import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
# esse processo envolve alguns passos como: 1.preparar os dados, 2. definir o modelo, 3. compilar o modelo, 4. treinar o modelo, 5. avaliar o modelo treinado e salvar.
# carregar dataset MNIST
(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()
# normalizar
x_train_full = x_train_full.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# reorganização e adicionamento 
x_train_full = np.expand_dims(x_train_full, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Split, vamos dividir o conjunto em uma parcela ainda menor para validação, que será usada para monitorar o desempenho do modelo durante o treinamento. Aqui, vamos usar 15% dos dados de treinamento para validação.
VAL_SPLIT = 0.15 
val_size = int(len(x_train_full) * VAL_SPLIT) # 
rng = np.random.default_rng(seed=42) # numero aleatório para reprodutibilidade
indices = rng.permutation(len(x_train_full))
val_idx, train_idx = indices[:val_size], indices[val_size:]
x_train, y_train = x_train_full[train_idx], y_train_full[train_idx]
x_val, y_val = x_train_full[val_idx], y_train_full[val_idx]

# CNN: 3 blocos Conv2D + BatchNorm + MaxPooling 
# definindo modelo
model = keras.Sequential([ #criando o modelo em sí , aqui sendo feito na forma de lista por ser uma forma mais facil de visualizar a arquitetura do modelo, mas poderia ser feito de outras formas(ex:.add).
    layers.Input(shape=(28, 28, 1)), # formato de entrada das imagens do MNIST
    layers.Conv2D(32, (3, 3), padding="same", activation="relu"), # 1 bloco, 32 filtros, kernel 3x3, padding "same" para manter o tamanho da imagem, função de ativação ReLU
    layers.BatchNormalization(), # função de normalização para melhorar a estabilidade e desempenho do modelo
    layers.MaxPooling2D((2, 2)), # função de pooling para reduzir a dimensionalidade da imagem, mantendo as características mais importantes
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"), # 2 bloco
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), padding="same", activation="relu"), # 3 bloco
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(), # achatamento da saída da última camada de pooling para uma dimensão 1D, preparando para a camada densa, aqui tudo vira uma grande lista
    layers.Dense(128, activation="relu"), # camada densa com 128 neurônios e função de ativação ReLU
    layers.Dropout(0.4), # dropout para reduzir os neuronios, desativando aleatoriamente 40% dos neurônios durante o treinamento
    layers.Dense(10, activation="softmax") # camada de saída com 10 neurônios (uma para cada classe) e função de ativação softmax para classificação multi-classe
])
# compilação do modelo(otimização, classificação e métricas)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# resumo do modelo
model.summary()

# treino com EarlyStopping
# parada antecipada para evitar overfitting, monitorando a perda de validação e restaurando os melhores pesos
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
# treinamento propriamente dito
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=15, # ciclo de treinamento, o modelo passa por todo o conjunto de dados de treinamento uma vez
    batch_size=64, # quantidade de amostras processadas antes de atualizar os pesos do modelo, aqui 64 amostras por vez
    callbacks=[early_stop], # aqui estamos passando a função de parada antecipada para o treinamento, para que ele pare se a perda de validação não melhorar por 3 épocas consecutivas
    verbose=2 # e aqui estamos definindo o nível de verbosidade, 2 significa que vamos ver uma barra de progresso para cada época, mostrando a perda e a acurácia de treinamento e validação
)

# Acurácia final e salvamento
val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
print(f"\n>>> Acuracia final de validacao: {val_acc:.4f}")
print(f">>> Loss final de validacao: {val_loss:.4f}")

model.save("model.h5")
print("\nModelo salvo em model.h5")