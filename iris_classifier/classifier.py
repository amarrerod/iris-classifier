

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class KNN:
    def __init__(self, dataset, x1 = None, x2 = None, y1 = None, y2 = None):
        self.data = dataset
        self.x = dataset.data
        self.y = dataset.target
        self.split_data()
        self.neighbors = 1
        self.model = KNeighborsClassifier(n_neighbors = self.neighbors)
    
    def split_data(self):
        self.x1, self.x2, self.y1, self.y2 = train_test_split(self.x, self.y, random_state = 0, train_size = 0.5)
    
    def train(self, x = None, y = None):
        self.model.fit(x, y)
    
    def predict(self, target):
        self.y_model = self.model.predict(target)
        return self.y_model
        
    def score(self, target):
        return accuracy_score(target, self.y_model)
    
    def holdout_validation(self):
        self.train(self.x1, self.y1)
        self.y2_model = self.model.predict(self.x2)
        return accuracy_score(self.y2, self.y2_model)
