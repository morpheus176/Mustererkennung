#pylint: disable=missing-docstring
import unittest
import numpy as np
from common.classification import KNNClassifier, GaussianClassifier


class ClassificationTest(unittest.TestCase):
    #
    # ACHTUNG: Die folgenden Tests sollen dazu dienen, das Verhalten der
    # Klassifikatoren anhand von einem kleinen Beispiel im Debugger
    # nachzuvollziehen.
    #
    # Der Erfolg oder Misserfolg der Tests haengt ausschliesslich von den
    # Klassifikationsergebnissen der 5 Test Samples ab.
    #
    # Da das Beispiel sehr klein ist, ist die Aussagekraft fuer die Korrektheit
    # Ihrer Implementierung begrenzt.

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.__train_samples = np.array([[2, 5, 0],
                                         [4, 1, 0],
                                         [3, 3, 1],
                                         [9, 8, 2],
                                         [1, 5, 3],
                                         [0, 7, 9],
                                         [1, 6, 8],
                                         [2, 9, 6],
                                         [0, 2, 3],
                                         [5, 3, 3]])
        self.__test_samples = np.array([[5, 0, 0],
                                        [0, 5, 0],
                                        [0, 0, 5],
                                        [5, 5, 0],
                                        [0, 5, 5]])
        self.__train_labels = np.array(['A', 'A', 'C', 'A', 'C',
                                        'B', 'B', 'B', 'C', 'C' ])

    def test_nn(self):
        print('nn_test')
        nn = KNNClassifier(k_neighbors=1, metric='cityblock')
        nn.estimate(self.__train_samples, self.__train_labels)
        result_labels = nn.classify(self.__test_samples)
        result_labels_ref = np.array(['A', 'A', 'C', 'A', 'C' ])
        self.assertEqual(result_labels_ref.shape, result_labels.shape)
        self.assertEqual(result_labels_ref.dtype, result_labels.dtype)
        np.testing.assert_equal(result_labels, result_labels_ref)

    def test_knn(self):
        print('knn_test')
        knn = KNNClassifier(k_neighbors=3, metric='cityblock')
        knn.estimate(self.__train_samples, self.__train_labels)
        result_labels = knn.classify(self.__test_samples)
        result_labels_ref = np.array(['C', 'C', 'C', 'A', 'C' ])
        self.assertEqual(result_labels_ref.shape, result_labels.shape)
        self.assertEqual(result_labels_ref.dtype, result_labels.dtype)
        np.testing.assert_equal(result_labels, result_labels_ref)

    def test_gauss(self):
        print('gauss_test')
        bayes = GaussianClassifier()
        bayes.estimate(self.__train_samples, self.__train_labels)
        result_labels = bayes.classify(self.__test_samples)
        result_labels_ref = np.array(['B', 'B', 'A', 'B', 'A' ])
        self.assertEqual(result_labels_ref.shape, result_labels.shape)
        self.assertEqual(result_labels_ref.dtype, result_labels.dtype)
        np.testing.assert_equal(result_labels, result_labels_ref)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
