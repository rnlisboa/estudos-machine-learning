from sklearn.model_selection import train_test_split

class DatasetSplitter:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def split_train_test(self, test_size=0.3, random_state=42):
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
