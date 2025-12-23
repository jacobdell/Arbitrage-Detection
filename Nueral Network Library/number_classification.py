import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('mnist_train.csv')
data = np.array(data)
m, n = data.shape

np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.


data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
Y_train_oh = one_hot(Y_train)
Y_dev_oh = one_hot(Y_dev)

class Module:
    def sgd_step(self, lrate): pass  # For modules w/o weights

class Linear(Module):
    def __init__(self, m, n):
        # initializes the weights randomly and offsets as 0
        self.m, self.n = (m, n)  # (in size, out size)
        self.W0 = np.zeros([self.n, 1])  # (n x 1)
        self.W = np.random.normal(0, 1.0 * m ** (-.5), [m, n])  # (m x n)

    def forward(self, A):
        # store the input matrix for future use
        self.A = A   # (m x b)  Hint: make sure you understand what b stands for
        Z = self.W.T@A +self.W0
        return Z

    def backward(self, dLdZ):
        # dLdZ is (n x b), uses stored self.A
        # store the derivatives for use in sgd_step and returd dLdA
        A = self.A
        self.dLdW = A @ dLdZ.T
        self.dLdW0 = np.sum(dLdZ, axis = 1, keepdims = True) 
        return self.W @ dLdZ

    def sgd_step(self, lrate):  # Gradient descent step
        self.W   -= lrate * self.dLdW
        self.W0    -= lrate*self.dLdW0      

class Sigmoid(Module):            # Layer activation
    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-Z))      
        return self.A

    def backward(self, dLdA):    
        return dLdA * self.A * (1-self.A)   
class ReLU(Module):              # Layer activation
    def forward(self, Z):
        self.A = np.maximum(0,Z)           # Your code
        return self.A

    def backward(self, dLdA):    # uses stored self.A
        return dLdA * (self.A > 0)
class SoftMax(Module):           # Output activation
    def forward(self, Z):
        Z_shift = Z - np.max(Z, axis=0, keepdims=True)  # subtract max for stability
        expZ = np.exp(Z_shift)
        self.A = expZ / np.sum(expZ, axis=0, keepdims=True)
        return self.A        

    def backward(self, dLdZ): 
        return dLdZ

    def class_fun(self, Ypred):
        # Returns the index of the most likely class for each point as vector of shape (b,)
        return np.argmax(Ypred, axis=0)# Your code

class NLL(Module):       # Loss
    def forward(self, Ypred, Y):
        # returns loss as a float
        self.Ypred = Ypred
        self.Y = Y
        loss = -np.sum(Y *np.log(Ypred))  
        return loss# Your code

    def backward(self):  # Use stored self.Ypred, self.Y
        # note, this is the derivative of loss with respect to the input of softmax
        return self.Ypred - self.Y  
class MSE(Module):
    def forward(self, Ypred, Y):
        """
        Compute mean squared error loss.
        Ypred and Y should have shape (output_dim, batch_size)
        """
        self.Ypred = Ypred
        self.Y = Y
        loss = np.mean((Ypred - Y) ** 2)
        return loss

    def backward(self):
        """
        Derivative of MSE with respect to input (dLoss/dYpred)
        """
        batch_size = self.Y.shape[1]
        return 2 * (self.Ypred - self.Y) / batch_size    
class Sequential:
    def __init__(self, modules, loss):            # List of modules, loss module
        self.modules = modules
        self.loss = loss

    def sgd(self, X, Y, iters=100, lrate=0.005):  # Train
        D, N = X.shape
        for it in range(iters):
            j = np.random.randint(N)
            Xt = X[:, j:j+1]
            Yt = Y[:, j:j+1]
            Ypred = self.forward(Xt)
            cur_loss = self.loss.forward(Ypred, Yt)
            delta = self.loss.backward()
            self.backward(delta)
            self.sgd_step(lrate)

    def forward(self, Xt):                        # Compute Ypred
        for m in self.modules: Xt = m.forward(Xt)
        return Xt

    def backward(self, delta):                    # Update dLdW and dLdW0
        # Note reversed list of modules
        for m in self.modules[::-1]: delta = m.backward(delta)

    def sgd_step(self, lrate):                    # Gradient descent step
        for m in self.modules: m.sgd_step(lrate)

    def print_accuracy(self, it, X, Y, cur_loss, every=250):
        # Utility method to print accuracy on full dataset, should
        # improve over time when doing SGD. Also prints COURSE/questions/nn loss,
        # which should decrease over time. Call this on each iteration
        # of SGD!
        if it % every == 1:
            cf = self.modules[-1].class_fun
            acc = np.mean(cf(self.forward(X)) == cf(Y))
            print('Iteration =', it, '	Acc =', acc, '	Loss =', cur_loss)

# Input size = 784, hidden = 128, output = 10
net = Sequential(
    modules=[
        Linear(784, 128),
        ReLU(),
        Linear(128, 10),
        SoftMax()
    ],
    loss=MSE()
)

net.sgd(X_train, Y_train_oh, iters=10000, lrate=0.001)

Ypred = net.forward(X_dev)
pred_classes = net.modules[-1].class_fun(Ypred)
accuracy = np.mean(pred_classes == Y_dev)
print("Dev set accuracy:", accuracy)
