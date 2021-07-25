import numpy as np

class NN():
    def __init__(self, inp, hidden, out):
        self.layers = [inp] + hidden + [out]
        self.activations = [np.zeros((1,l)) for l in self.layers]
        self.weights = [(2*np.random.random((self.layers[i], self.layers[i+1]))-1) for i in range(len(self.layers)-1)]
        self.derivatives = self.weights.copy()
    
    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def _sigmoid_p(self, s):
        return s * (1 - s)

    def forward(self, inputs):
        self.activations[0] = np.array(inputs)
        for i,w in enumerate(self.weights):
            self.activations[i+1] = self._sigmoid(np.dot(self.activations[i], w))
        return self.activations[-1]

    def backward(self, error):
        for i in reversed(range(len(self.weights))):
            delta = error * self._sigmoid_p(self.activations[i+1])
            self.derivatives[i] = np.dot(self.activations[i].T, delta)
            error = np.dot(delta, self.weights[i].T)

        for i, d in enumerate(self.derivatives):
            self.weights[i] += d

        return error

    def train(self, tins, touts, iters):
        for k in range(iters):
            e = touts - self.forward(tins)
            self.backward(e)
            # for i, d in enumerate(self.derivatives):
            #     self.weights[i] += d

# n = NN(3,[100],1)
# n.train([[1,1,1],[1,0,1],[0,0,0],[0,1,0]], [[1],[1],[0],[0]], 2000)

