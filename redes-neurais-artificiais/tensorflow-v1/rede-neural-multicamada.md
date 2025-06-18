
# Classificação de Flores Iris com MLP (TensorFlow)

## Objetivo do Projeto

O objetivo deste projeto é realizar a **classificação de flores da base de dados Iris**, utilizando uma **Rede Neural MLP (Perceptron Multicamadas)** implementada com **TensorFlow**. Este trabalho é um exemplo prático de aplicação de redes neurais para problemas de classificação multiclasse.

## Descrição do Projeto

A base **Iris** contém amostras de três tipos de flores: *Iris setosa*, *Iris versicolor* e *Iris virginica*. Cada amostra possui quatro atributos numéricos referentes a medidas físicas das flores.

O projeto envolve as seguintes etapas principais:

---

## Etapas Detalhadas

### 1. Importação e carregamento da base Iris

```python
from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target
```

- **X** contém as características das flores (entradas da rede)
- **y** contém os rótulos das classes (0, 1, 2)

### 2. Normalização dos dados de entrada (X)

```python
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)
```

> Normalizar ajuda a acelerar o treinamento e melhorar o desempenho.

### 3. Codificação One-Hot dos rótulos (y)

```python
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder

onehot = ColumnTransformer([("OneHot", OneHotEncoder(), [0])], remainder="passthrough")
y = y.reshape(-1, 1)
y = onehot.fit_transform(y)
```

> Convertemos os rótulos (0, 1, 2) em formato One-Hot:  
Exemplo:  
Classe 0 → [1, 0, 0]  
Classe 1 → [0, 1, 0]  
Classe 2 → [0, 0, 1]

### 4. Separação em treino e teste

```python
from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size=0.3)
```

> Utilizamos 70% para treino e 30% para teste.

### 5. Definição da estrutura da MLP

```python
import tensorflow as tf
import numpy as np

neuronios_entrada = X.shape[1]
neuronios_oculta = int(np.ceil((X.shape[1] + y.shape[1])) / 2)
neuronios_saida = y.shape[1]
```

- **Número de neurônios de entrada:** igual ao número de features (4)
- **Número de neurônios na camada oculta:** média entre entrada e saída
- **Número de neurônios de saída:** igual ao número de classes (3)

### 6. Inicialização dos pesos e biases

```python
W = {
    'oculta': tf.Variable(tf.random_normal([neuronios_entrada, neuronios_oculta])),
    'saida': tf.Variable(tf.random_normal([neuronios_oculta, neuronios_saida]))
}

b = {
    'oculta': tf.Variable(tf.random_normal([neuronios_oculta])),
    'saida': tf.Variable(tf.random_normal([neuronios_saida]))
}
```

### 7. Placeholders para entrada e saída

```python
xph = tf.placeholder('float', [None, neuronios_entrada])
yph = tf.placeholder('float', [None, neuronios_saida])
```

> Placeholders para inserir os dados durante o treinamento.

### 8. Função para construção da MLP

```python
def mlp(x, W, bias):
    camada_oculta = tf.add(tf.matmul(x, W['oculta']), bias['oculta'])
    camada_oculta_ativacao = tf.nn.relu(camada_oculta)
    camada_saida = tf.add(tf.matmul(camada_oculta_ativacao, W['saida']), bias['saida'])
    return camada_saida
```

- Camada oculta com ativação ReLU
- Camada de saída sem ativação (logits)

### 9. Função de custo e otimizador

```python
modelo = mlp(xph, W, b)
erro = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=modelo, labels=yph))
otimizador = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(erro)
```

> Utilizamos **softmax cross-entropy** como função de perda.  
O otimizador escolhido foi o **Adam**, com taxa de aprendizado de 0.0001.

### 10. Treinamento em batches

```python
batch_size = 8
batch_total = int(len(X_treinamento) / batch_size)
X_batches = np.array_split(X_treinamento, batch_total)
```

> Dividimos o conjunto de treino em mini-batches para melhor eficiência.

### 11. Loop de treinamento

```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoca in range(20000):
        erro_medio = 0
        batch_total = int(len(X_treinamento) / batch_size)
        X_batches = np.array_split(X_treinamento, batch_total)
        y_batches = np.array_split(y_treinamento, batch_total)
        for i in range(batch_total):
            X_batch, y_batch = X_batches[i], y_batches[i]
            _, custo = sess.run([otimizador, erro], feed_dict={xph: X_batch, yph: y_batch})
            erro_medio = custo / batch_total
        if epoca % 500 == 0:
            print(f"Época {epoca + 1} erro {erro_medio}")
    W_final, b_final = sess.run([W, b])
```

> Treinamos por 20000 épocas, exibindo o erro médio a cada 500 iterações.

### 12. Teste do modelo treinado

```python
previsoe_teste = mlp(xph, W_final, b_final)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    r1 = sess.run(previsoe_teste, feed_dict={xph: X_teste})
    r2 = sess.run(tf.nn.softmax(r1))
    r3 = sess.run(tf.argmax(r2, 1))
```

> Calculamos as predições e aplicamos a função **softmax** seguida do **argmax** para obter as classes previstas.

### 13. Avaliação de desempenho

```python
from sklearn.metrics import accuracy_score

y_teste2 = np.argmax(y_teste, 1)
taxa_acerto = accuracy_score(y_teste2, r3)
print(taxa_acerto)
```

> Avaliamos a **taxa de acerto (acurácia)** comparando os rótulos reais com os previstos.

---

## Conclusão

Este projeto demonstrou a aplicação de um **MLP simples implementado com TensorFlow 1.x** para um problema clássico de classificação. O código abrange o fluxo completo: desde o pré-processamento até a avaliação final.

---

## Requisitos do Projeto

- Python 3.x
- TensorFlow 1.x
- Scikit-learn
- NumPy

---

## Autor

Projeto acadêmico desenvolvido por Renan Lisboa.
