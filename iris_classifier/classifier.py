

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class KNN:
    def __init__(self, dataset):
        self.data = dataset
        self.x = dataset.data
        self.y = dataset.target