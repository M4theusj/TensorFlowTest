import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # diminui os outputs do tf

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# dataset MNIST é um conjunto de imagens de números escritos a mão
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normaliza os valores dos pixels para o intervalo [0,1] e achata as imagens de 28x28 para um vetor de 784 elementos
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

# Sequential API (mais simples, mas menos flexível, aceita apenas uma entrada e uma saída)
# model = keras.Sequential(
#     [
#         keras.Input(shape=(28*28,)),  # define a camada de entrada com 784 neurônios (28x28)
#         layers.Dense(512, activation='relu'),  # primeira camada oculta com 512 neurônios
#         layers.Dense(256, activation='relu'),  # segunda camada oculta com 256 neurônios
#         layers.Dense(10),  # camada de saída com 10 neurônios (um para cada dígito)
#     ]
# ) relu "elimina" números negativos os transformando em zero

# Functional API (mais flexível e modular)
inputs = keras.Input(shape=(784,))  # define a camada de entrada
x = layers.Dense(512, activation='relu', name="first_layer")(inputs)  # primeira camada oculta com 512 neurônios e ReLU
x = layers.Dense(256, activation='relu', name="second_layer")(x)  # segunda camada oculta com 256 neurônios
outputs = layers.Dense(10, activation='softmax')(x)  # camada de saída com 10 neurônios e ativação softmax
model = keras.Model(inputs=inputs, outputs=outputs)  # define o modelo com entrada e saída

# print(model.summary())  # exibe a estrutura do modelo

# compila o modelo definindo a função de perda, o otimizador e a métrica de avaliação
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # função de perda para classificação categórica
    optimizer=keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9),  # otimizador Adam com taxa de aprendizado 0.001
    metrics=["accuracy"],  # mede a acurácia do modelo
)

# treina o modelo com os dados de treino
model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)

# avalia o modelo nos dados de teste
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
