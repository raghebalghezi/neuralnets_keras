import math
from typing import List

import numpy as np


class SimpleNetwork:
    """A simple feedforward network with a single hidden layer. All units in
    the network have sigmoid activation.

    """

    @classmethod
    def of(cls, n_input: int, n_hidden: int, n_output: int):
        """Creates a single-layer feedforward neural network with the given
        number of input, hidden, and output units.

        :param n_input: Number of input units
        :param n_hidden: Number of hidden units
        :param n_output: Number of output units
        :return: the neural network
        """

        def uniform(n_in, n_out):
            epsilon = math.sqrt(6) / math.sqrt(n_in + n_out)
            return np.random.uniform(-epsilon, +epsilon, size=(n_in, n_out))

        return cls(uniform(n_input, n_hidden), uniform(n_hidden, n_output))
    

    def __init__(self,
                 input_to_hidden_weights: np.ndarray,
                 hidden_to_output_weights: np.ndarray):
        """Creates a neural network from two weights matrices, one representing
        the weights from input units to hidden units, and the other representing
        the weights from hidden units to output units.

        :param input_to_hidden_weights: The weight matrix mapping from input
        units to hidden units
        :param hidden_to_output_weights: The weight matrix mapping from hiddden
        units to output units
        """
        self.input_to_hidden_weights = input_to_hidden_weights
        self.hidden_to_output_weights = hidden_to_output_weights

#        self.input_to_hidden_weights = np.random.randn(self.n_input,self.n_hidden)
#        self.hidden_to_output_weights = np.random.randn(self.n_hidden,self.n_output)
        
    def sigmoid(self,s):
    # activation function
        return 1/(1+np.exp(-s))
    
    def sigmoid_deriv(self,s):
    # the derivative of the activation function
        return self.sigmoid(s) * (1 - self.sigmoid(s))

    def predict(self, input_matrix: np.ndarray,flag=False) -> np.ndarray:
        """Performs forward propagation over the neural network starting with
        the given input matrix.

        Each unit's output should be calculated by taking a weighted sum of its
        inputs (using the appropriate weight matrix) and passing the result of
        that sum through a logistic sigmoid activation function.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :return: A matrix of predictions, where each row is the predicted
        outputs - each in the range (0, 1) - for the corresponding row in the
        input matrix.
        """

        self.first_layer = np.matmul(input_matrix,self.input_to_hidden_weights)
        self.first_layer_a = self.sigmoid(self.first_layer)
        #multiply first_layer by corresponding weights wrapped by sigmoid activation
        self.hidden_layer = np.matmul(self.first_layer_a,self.hidden_to_output_weights)
        self.hidden_layer_a = self.sigmoid(self.hidden_layer)
        if flag:
            return self.first_layer, self.first_layer_a, self.hidden_layer, self.hidden_layer_a
        else:
            return self.hidden_layer_a 
        #               h_1               a_1                 h_2                  a_2

    def predict_zero_one(self, input_matrix: np.ndarray) -> np.ndarray:
        """Performs forward propagation over the neural network starting with
        the given input matrix, and converts the outputs to binary (0 or 1).

        Outputs will be converted to 0 if they are less than 0.5, and converted
        to 1 otherwise.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :return: A matrix of predictions, where each row is the predicted
        outputs - each either 0 or 1 - for the corresponding row in the input
        matrix.
        """
        res = self.predict(input_matrix)
        lst = (res >= 0.5).astype(int)
        return lst

    def gradients(self,
                  input_matrix: np.ndarray,
                  output_matrix: np.ndarray) -> List[np.ndarray]:
        """Performs back-propagation to calculate the gradients for each of
        the weight matrices.

        This method first performs a pass of forward propagation through the
        network, keeping track of the weighted sums before the activation
        function (at layer l, we call such a vector a_l) and the values
        after the activation function (at layer l, we call such a vector h_l,
        and similarly refer to the input as h_0).

        Then for each input example, the method applies the following
        calculations, where × is matrix multiplication, ⊙ is element-wise
        product, and ⊤ is matrix transpose:

        1. g_2 = ((h_2 - y) ⊙ sigmoid'(a_2))⊤ // g_2 = np.transpose((h_2 - output_matrix) * self.sigmoid_deriv(a_2))
        2. hidden-to-output weights gradient += (g_2 × h_1)⊤
        3. g_1 = (((hidden-to-output weights) × g_2)⊤ ⊙ sigmoid'(a_1))⊤
        4. input-to-hidden weights gradient += (g_1 × h_0)⊤

        When all input examples have applied their updates to the gradients,
        each entire gradient should be divided by the number of input examples.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :param output_matrix: A matrix of expected outputs, where each row is
        the expected outputs - each either 0 or 1 - for the corresponding row in
        the input matrix.
        :return: two matrices of gradients, one for the input-to-hidden weights
        and one for the hidden-to-output weights
        """
        # Using the flag, the predict method returns h_1,a_1,h_2,a_2
        h_1,a_1,h_2,a_2 = self.predict(input_matrix,flag=True) # perform foward pass
        g_2 = np.transpose((a_2 - output_matrix) * self.sigmoid_deriv(h_2)) #Error of output layer
        h2ow_gradient = np.transpose(np.matmul(g_2,a_1)) # Cost derivative for h2o weight
        h2ow_gradient /= input_matrix.shape[0] #gradient divided by the number of input examples
        g_1 = np.transpose(np.matmul(self.hidden_to_output_weights,g_2).T * self.sigmoid_deriv(h_1)) #Error of hidden layer
        i2hw_gradient = np.transpose(np.matmul(g_1,input_matrix)) # Cost derivative for i2h weight
        i2hw_gradient /= input_matrix.shape[0] #gradient divided by the number of input examples

        return i2hw_gradient, h2ow_gradient
        
    def train(self,
              input_matrix: np.ndarray,
              output_matrix: np.ndarray,
              iterations: int = 10,
              learning_rate: float = 0.1) -> None:
        """Trains the neural network on an input matrix and an expected output
        matrix.

        Training should repeatedly (`iterations` times) calculate the gradients,
        and update the model by subtracting the learning rate times the
        gradients from the model weight matrices.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :param output_matrix: A matrix of expected outputs, where each row is
        the expected outputs - each either 0 or 1 - for the corresponding row in
        the input matrix.
        :param iterations: The number of gradient descent steps to take.
        :param learning_rate: The size of gradient descent steps to take, a
        number that the gradients should be multiplied by before updating the
        model weights.
        """
        
        for i in range(iterations):
            i2hw,h2ow  = self.gradients(input_matrix,output_matrix)
            self.input_to_hidden_weights -= learning_rate * i2hw
            self.hidden_to_output_weights -= learning_rate * h2ow
        
            