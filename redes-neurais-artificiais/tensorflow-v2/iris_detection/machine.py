import numpy as np
import tensorflow as tf

class Machine:
    def __init__(self, X_train, y_train):
        self.input_dim = None
        self.output_dim = None
        self.hidden_dim = None
        self.X_train = X_train
        self.y_train = y_train
        self.model = None
        
    def create_neural_network_arch(self, inputs_perceptron, outs_perceptron):
        self.input_dim = inputs_perceptron
        self.output_dim = outs_perceptron
        self.hidden_dim = int(np.ceil((self.input_dim + self.output_dim) / 2))

    def build_iris_model_predict(self):
        self.model = tf.keras.Sequential([
        tf.keras.layers.Dense(self.hidden_dim, activation='relu', input_shape=(self.input_dim,)),
        tf.keras.layers.Dense(self.output_dim, activation='softmax')
    ])
        
    def compile_model(self):
        self.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
        
    def learn(self):
        print("A máquina está aprendendo...")
        self.model.fit(
        self.X_train,
        self.y_train,
        epochs=200,
        batch_size=8,
        verbose=2
    )
        print("A máquina aprendeu!!!")
    
    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        return loss, accuracy
    
    def get_model(self):
        return self.model