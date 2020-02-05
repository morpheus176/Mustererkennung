import unittest

import torch.nn as nn
from common.cnn import ConvolutionalNeuralNetwork


class Test(unittest.TestCase):


    def testArchitecture(self):
        cnn = ConvolutionalNeuralNetwork()
        names = ['conv1', 'conv2', 'fc1', 'fc2']
        sizes = [(10, 1, 5, 5), (20, 10, 5, 5), (10, 320), (10, 10)]
        child_modules = [module for module in cnn.children() if not isinstance(module, (nn.ReLU, nn.MaxPool2d))]
        self.assertEqual(len(child_modules), 4, msg='Anzahl der Schichten (ausser ReLU und MaxPool2d ist fehlerhaft')
        for name, size, module in zip(names, sizes, child_modules):
            self.assertSequenceEqual(module.weight.size(), size, msg='Weihts of %s should have shape (%s)' % \
                                     (name, ', '.join([str(i) for i in size])))

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
