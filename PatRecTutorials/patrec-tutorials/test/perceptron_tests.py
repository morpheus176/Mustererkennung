import unittest
import numpy as np
from common.classification import NeuralNetwork

class Test(unittest.TestCase):

    def testActivation(self):
        x_input = np.load('neural_network_data/x_input.npy')
        expected_activations = np.load('neural_network_data/expected_activations.npy')
        expected_activation_derivs = np.load('neural_network_data/expected_activation_derivatives.npy')
        exp_linear_output, exp_tanh_output, exp_sigmoid_output = np.split(expected_activations, 3 , 0)
        exp_linear_deriv, exp_tanh_deriv, exp_sigmoid_deriv = np.split(expected_activation_derivs, 3 , 0)
        nn = NeuralNetwork(learning_rate=1, iterations=1, activation='Linear')
        linear_output = nn.activation(x_input)
        linear_deriv = nn.activation_deriv(x_input)
        np.testing.assert_equal(actual=linear_output,
                                desired=x_input,
                                err_msg='Die lineare Aktivierungsfunktion ist fehlerhaft.')
        np.testing.assert_equal(actual=linear_deriv,
                                desired=np.ones_like(x_input),
                                err_msg='Die Ableitung der linearen Aktivierungsfunktion ist fehlerhaft.')
        nn = NeuralNetwork(learning_rate=1, iterations=1, activation='TanH')
        tanh_output = nn.activation(x_input)
        tanh_deriv = nn.activation_deriv(x_input)
        np.testing.assert_equal(actual=tanh_output,
                                desired=exp_tanh_output,
                                err_msg='Die TanH-Aktivierungsfunktion ist fehlerhaft.')
        np.testing.assert_equal(actual=tanh_deriv,
                                desired=exp_tanh_deriv,
                                err_msg='Die Ableitung der TanH-Aktivierungsfunktion ist fehlerhaft.')
        tanh_output = nn.activation(x_input)
        nn = NeuralNetwork(learning_rate=1, iterations=1, activation='Sigmoid')
        sigmoid_output = nn.activation(x_input)
        sigmoid_deriv = nn.activation_deriv(x_input)
        np.testing.assert_equal(actual=sigmoid_output,
                                desired=exp_sigmoid_output,
                                err_msg='Die Sigmoid-Aktivierungsfunktion ist fehlerhaft.')
        np.testing.assert_equal(actual=sigmoid_deriv,
                                desired=exp_sigmoid_deriv,
                                err_msg='Die Ableitung der Sigmoid-Aktivierungsfunktion ist fehlerhaft.')
        tanh_output = nn.activation(x_input)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
