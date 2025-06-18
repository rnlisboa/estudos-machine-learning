from sklearn import datasets
from sklearn.discriminant_analysis import StandardScaler

class IrisDataset:

    def __init__(self):
        iris = datasets.load_iris()
        self.X = iris.data
        self.y = iris.target

    def get_dataset(self, ):
        return self.X, self.y
