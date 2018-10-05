

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class KNN:
    def __init__(self, dataset):
        self.data = dataset
        self.x = dataset.data
        self.y = dataset.target
        self.neighbors = 1
        self.model = KNeighborsClassifier(n_neighbors = self.neighbors)

    def train(self):
        self.model.fit(self.x, self.y)
    