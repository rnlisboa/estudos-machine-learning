import numpy as np

from sklearn.metrics import accuracy_score

from normalize_dataset import NormalizeDataset
from dataset_splitter import DatasetSplitter
from load_iris_dataset import IrisDataset
from machine import Machine

iris_dataset = IrisDataset()

normalizer_dataset = NormalizeDataset(iris_dataset)
dataset_splitter = DatasetSplitter(normalizer_dataset.scaler_iris_data(), normalizer_dataset.scaler_target_data())

X_train, X_test, y_train, y_test = dataset_splitter.split_train_test()

if __name__=="__main__":
    machine = Machine(X_train=X_train, y_train=y_train)
    machine.create_neural_network_arch(inputs_perceptron=iris_dataset.X.shape[1], outs_perceptron=len(np.unique(iris_dataset.y.reshape(-1, 1))))
    machine.build_iris_model_predict()
    machine.compile_model()

    machine.learn()

    loss, accuracy = machine.evaluate(X_test, y_test)
    print(f"\nAcurácia no teste: {accuracy:.4f}")


    y_pred_probs = machine.get_model().predict(X_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)


    sklearn_accuracy = accuracy_score(y_test_classes, y_pred_classes)
    print(f"Acurácia calculada manualmente: {sklearn_accuracy:.4f}")
