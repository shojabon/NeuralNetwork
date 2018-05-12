import base64
from collections import OrderedDict

import numpy as np

from neuralnetwork.st_layer import st_layer
from neuralnetwork.st_layer_function.Affine import Affine
from neuralnetwork.st_layer_function.Relu import Relu
from neuralnetwork.st_layer_function.Softlost import Softlost


class st_model:

    def __init__(self, input_size, hidden_size, output_size, starting_weights=0.01):
        self.weights = OrderedDict()
        self.model = OrderedDict()
        self.optimizers = []
        self.layers = []
        self.forward_functions = []
        self.model_shape = []
        self.model_shape.append(input_size)
        self.model_shape.extend(hidden_size)
        self.model_shape.append(output_size)
        self.weights['W1'] = starting_weights * np.random.rand(input_size, hidden_size[0])
        self.weights['b1'] = np.zeros(hidden_size[0])
        i = 2
        for x in range(0, len(hidden_size) - 1):
            self.weights['W' + str(i)] = starting_weights * np.random.rand(hidden_size[x], hidden_size[x+1])
            self.weights['b' + str(i)] = np.zeros(hidden_size[x+1])
            i += 1
        self.weights['W' + str(i)] = starting_weights * np.random.rand(hidden_size[int(len(hidden_size))-1], output_size)
        self.weights['b' + str(i)] = np.zeros(output_size)
        self.last = Softlost()
        for x in range(0, int(len(self.weights.keys())/2)):
            self.model['layer' + str(x + 1)] = Affine(self.weights['W' + str(x + 1)], self.weights.get('b' + str(x + 1)))
            self.model['relu' + str(x + 1)] = Relu()

    def get_model(self):
        return self.model

    def get_answer(self, input):
        x = self.predict(input)
        return self.softmax(x)

    def softmax(self, x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T

        x = x - np.max(x)
        return np.exp(x) / np.sum(np.exp(x))

    def get_loss(self, data, lable):
        out = self.predict(data)
        return self.last.forward(out, lable)

    def predict(self, x):
        for y in self.model.values():
            x = y.forward(x)
        return x

    def get_shape(self):
        return self.model_shape

    def get_accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def get_gradient(self, x, t):
        self.get_loss(x, t)
        dout = 1
        dout = self.last.backward(dout)
        layers = list(self.model.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        for x in range(0, int(len(self.model)/2)):
            grads['W' + str(x + 1)], grads['b' + str(x + 1)] = self.model['layer' + str(x + 1)].dW, self.model['layer' + str(x + 1)].db
        return grads

    def learn(self, x, t):
        gradient = self.get_gradient(x, t)
        for x in gradient.keys():
            self.weights[x] -= 0.1 * gradient[x]