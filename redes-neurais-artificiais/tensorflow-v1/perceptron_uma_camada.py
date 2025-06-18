import tensorflow as tf
import numpy as np

# Definindo os dados de entrada e saída
X = np.array([[0.0, 0.0], 
              [0.0, 1.0],
              [1.0, 0.0],
              [1.0, 1.0]])

y = np.array([[0.0], [0.0], [0.0], [1.0]])  # resultado esperado da operação

# Inicializando os pesos
qtd_linhas = 2
qtd_colunas = 1
w = tf.Variable(tf.zeros([qtd_linhas, qtd_colunas], dtype=tf.float64))  # pesos

init = tf.global_variables_initializer()

# Definindo a camada de saída e a função de ativação
camada_saida = tf.matmul(X, w)

def step(x):
    return tf.cast(tf.to_float(tf.math.greater_equal(x, 1)), tf.float64)  # caso true retorna 1.0, caso false retorna 0.0

camada_saida_ativacao = step(camada_saida)

# Calculando o erro e atualizando os pesos
erro = tf.subtract(y, camada_saida_ativacao)

delta = tf.matmul(X, erro, transpose_a=True)
treinamento = tf.assign(w, tf.add(w, tf.multiply(delta, 0.1)))

# Treinamento do perceptron
with tf.Session() as sess:
    sess.run(init)
    print('\n\n')
    epoca = 0
    for i in range(15):
        epoca += 1
        erro_total, _ = sess.run([erro, treinamento])
        erro_soma = tf.reduce_sum(erro_total)
        print(erro_total)
        print('Epoca: ', epoca, "Erro: ", sess.run(erro_soma))
        if erro_soma.eval() == 0.0:
            break
    w_final = sess.run(w)

# Exibindo os pesos finais
print("Pesos finais:")
print(w_final)

# Testando o perceptron treinado
camada_saida_teste = tf.matmul(X, w_final)
camada_saida_ativacao_teste = step(camada_saida_teste)
with tf.Session() as sess:
    sess.run(init)
    print("Saída do teste:")
    print(sess.run(camada_saida_ativacao_teste))