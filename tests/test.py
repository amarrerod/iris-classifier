
import unittest
from sklearn.datasets import load_iris
from iris_classifier.classifier import KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection._validation import cross_val_score

class Test_Classifier(unittest.TestCase):
    def setUp(self):
        self.iris = load_iris()
        self.x = self.iris.data
        self.y = self.iris.target
        self.x1, self.x2, self.y1, self.y2 = train_test_split(self.x, self.y, random_state = 0, train_size = 0.5)
        self.neighbors = 1
        self.classifier = KNN(self.iris, self.x1, self.x2, self.y1, self.y2, cross_validation = False, folds = 1)
        self.classifier.train(self.x, self.y)
        self.classifier.predict(self.x)

    def test_shape_data(self):
        self.assertEqual(self.x.shape,  self.classifier.x.shape)
    
    def test_shape_taget(self):
        self.assertEqual(self.y.shape, self.classifier.y.shape)

    def test_neighbors(self):
        self.assertEqual(self.neighbors, self.classifier.neighbors)
    
    def test_KNN(self):
        self.assertIsInstance(self.classifier.model, KNeighborsClassifier)

    def test_classifer_has_train_method(self):
        self.assertTrue(self.classifier.train)
    
    def test_predict_labels(self):
        self.assertTrue(self.classifier.predict)

    def test_training_score_equals_one(self):
        value = self.classifier.score(self.classifier.y)
        self.assertAlmostEqual(value, 1.0)
    
    def test_has_two_sets(self):
        self.assertEqual(self.x2.shape, self.classifier.x2.shape)

    def test_has_holdout_validation(self):
        self.assertTrue(self.classifier.holdout_validation())

    def test_holdout_validation_equal(self):
        self.classifier.train(self.x1, self.y1)
        y_model = self.classifier.model.predict(self.x2)
        expected = accuracy_score(self.y2, y_model)
        self.assertEqual(expected, self.classifier.holdout_validation())
    
    def test_cross_validation(self):
        y2_model = self.classifier.model.fit(self.x1, self.y1).predict(self.x2)
        y1_model = self.classifier.model.fit(self.x2, self.y2).predict(self.x1)
        expected = [accuracy_score(self.y1, y1_model), accuracy_score(self.y2, y2_model)]
        returned = self.classifier.cross_validation()
        self.assertEqual(expected[0], returned[0])
        self.assertEqual(expected[1], returned[1])

    def test_cross_validation_5_folds(self):
        model = KNeighborsClassifier(n_neighbors = self.neighbors)
        expected = cross_val_score(model, self.x, self.y, cv = 5)
        returned = self.classifier.cross_validation(5)
        for i in range(5):
            self.assertEqual(expected[i], returned[i])