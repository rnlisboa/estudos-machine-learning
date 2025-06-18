from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder



class NormalizeDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.onehot = OneHotEncoder(sparse_output=False)

    def get_dataset(self):
        X, y = self.dataset.get_dataset()
        return X, y
    
    def scaler_iris_data(self):
        scaler_x = StandardScaler()
        iris_data, _ = self.get_dataset()

        return scaler_x.fit_transform(iris_data)
    
    def scaler_target_data(self):
        """
        OneHotEncoder transforma os rótulos de classes inteiras (0, 1, 2) em vetores binários, como:

        - Classe 0 → [1, 0, 0]
        - Classe 1 → [0, 1, 0]
        - Classe 2 → [0, 0, 1]
        """
        _, y = self.get_dataset()
        return self.onehot.fit_transform(y.reshape(-1, 1))
    
    