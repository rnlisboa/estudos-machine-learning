# Classificação Binária com Rede Neural - Problema XOR

Este trabalho tem como objetivo demonstrar a implementação de uma rede neural simples para a resolução do problema clássico da lógica XOR utilizando a biblioteca TensorFlow (versão 1.x).

---

## 1. Bibliotecas Utilizadas

```python
import tensorflow as tf
import numpy as np
```

- **TensorFlow**: Framework para construção e treinamento de redes neurais.
- **NumPy**: Biblioteca para manipulação de vetores e matrizes.

---

## 2. Definição dos Dados

```python
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[1], [0], [0], [1]])  # resultado esperado da operação XOR
```

- `X`: Entradas da função XOR.
- `y`: Saídas esperadas, representando o comportamento da porta lógica XOR.

---

## 3. Arquitetura da Rede Neural

```python
neuronios_entrada = 2
neuronios_oculta = 3
neuronios_saida = 1
```

A rede é composta por:
- 2 neurônios na camada de entrada.
- 3 neurônios na camada oculta.
- 1 neurônio na camada de saída.

---

## 4. Inicialização dos Pesos e Bias

```python
W = {
    'oculta': tf.Variable(tf.random_normal([neuronios_entrada, neuronios_oculta]), name='w_oculta'),
    'saida': tf.Variable(tf.random_normal([neuronios_oculta, neuronios_saida]), name='w_saida')
}

b = {
    'oculta': tf.Variable(tf.random_normal([neuronios_oculta]), name='b_oculta'),
    'saida': tf.Variable(tf.random_normal([neuronios_saida]), name='b_saida')
}
```

Os pesos e bias são inicializados com valores aleatórios usando distribuição normal.

---

## 5. Placeholders

```python
xph = tf.placeholder(tf.float32, [4, neuronios_entrada], name='xph')
yph = tf.placeholder(tf.float32, [4, neuronios_saida], name='yph')
```

Usados para alimentar os dados de entrada (`X`) e saída (`y`) durante a execução.

---

## 6. Construção da Rede

```python
camada_oculta = tf.add(tf.matmul(xph, W['oculta']), b['oculta'])
camada_oculta_ativacao = tf.sigmoid(camada_oculta)

camada_saida = tf.add(tf.matmul(camada_oculta_ativacao, W['saida']), b['saida'])
camada_saida_ativacao = tf.sigmoid(camada_saida)
```

- Multiplicação das entradas com os pesos.
- Soma com os bias.
- Aplicação da função de ativação sigmoide.

---

## 7. Função de Erro e Otimizador

```python
erro = tf.losses.mean_squared_error(yph, camada_saida_ativacao)
otimizador = tf.train.GradientDescentOptimizer(learning_rate=0.3).minimize(erro)
```

- A função de erro usada é o erro quadrático médio.
- A otimização é feita com **Gradiente Descendente**.

---

## 8. Inicialização das Variáveis

```python
init = tf.global_variables_initializer()
```

Necessário antes de treinar a rede.

---

## 9. Treinamento da Rede

```python
with tf.Session() as sess:
    sess.run(init)
    for epocas in range(100000):
        _, custo = sess.run([otimizador, erro], feed_dict={xph: X, yph: y})
        if epocas % 200 == 0:
            print(custo / 4)
    W_final, b_final = sess.run([W, b])
```

- Treina a rede por 100 mil épocas.
- Mostra o erro médio a cada 200 iterações.
- Ao final, armazena os pesos e bias treinados.

---

## 10. Reconstrução da Rede com Pesos Finais

```python
camada_oculta_teste = tf.add(tf.matmul(xph, W_final['oculta']), b_final['oculta'])
camada_oculta_ativacao_teste = tf.sigmoid(camada_oculta_teste)

camada_saida_teste = tf.add(tf.matmul(camada_oculta_ativacao_teste, W_final['saida']), b_final['saida'])
camada_saida_ativacao_teste = tf.sigmoid(camada_saida_teste)
```

---

## 11. Avaliação da Rede

```python
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(camada_saida_ativacao_teste, feed_dict={xph: X}))
```

A saída deverá se aproximar de:

```
[[1], [0], [0], [1]]
```

---

## Conclusão

Este experimento demonstra a capacidade de uma rede neural simples em aprender a função XOR, um problema que não é linearmente separável. A camada oculta é essencial para permitir essa aprendizagem. O uso de TensorFlow proporciona uma abordagem estruturada e eficiente para esse tipo de tarefa.
