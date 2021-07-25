import numpy as np
from nn import NN

# n = NN(3,[100],1)
# n.train([[1,1,1],[1,0,1],[0,0,0],[0,1,0]], [[1],[1],[0],[0]], 2000)
LR = 0.00001

class Agent():
    def __init__(self, inp, hidden, out):
        self.out = out
        self.nn = NN(inp, hidden, out)
        self.actions = [i for i in range(out)]
        self.games = 0
        self.wins = 0

    def getActionGrad(self, action, reward):
        # return a gradient that only affects the specified action
        g = np.zeros((1,self.out))
        g[0][action] = reward
        return g

    def update(self, action, reward):
        grad = self.getActionGrad(action, reward)
        self.nn.backward(grad * LR)
        self.games += 1

    def choice(self, X):
        y = self.nn.forward(X)
        yflat = y.flatten()
        p = yflat / sum(yflat)
        action = np.random.choice(self.actions, p = p)
        return action, y, p
