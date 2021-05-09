import numpy as np


## activation functions ######
def linear(z, derivative=False):
    # for regression problems
    a = z
    if derivative:
        da = np.ones(z.shape)
        return a, da
    return a

def logistic(z, derivative=False):
    # for Multi-Label classification problems
    a = 1 / (1 + np.exp(-z))
    if derivative:
        da = np.ones(z.shape)     
        return a, da
    return a

def softmax(z, derivative=False):
    # for Multi-Class clasification problems
    e = np.exp(z - np.max(z, axis=0))
    a = e / np.sum(e, axis=0)
    if derivative:
        da = np.ones(z.shape)
        return a, da
    return a

class OLN:
    ''''One-Layer Network'''
    
    def __init__(self, n_inputs, n_outputs, activation_function=linear):
        '''
        Parameters
        ----------
        n_inputs : INT
            Problem dimension.
        n_outputs : INT
            Number of neurons.
        activation_function : choose activation fucntion: linear, sigmoid or softmax
            neural network activation function. The default is linear.
        Returns
        -------
        None.

        '''
        
        self.w = -1 + 2 * np.random.rand(n_outputs, n_inputs) # matrix of synaptic weights
        self.b = -1 + 2 * np.random.rand(n_outputs, 1)        # bias vector of every neuron
        self.f = activation_function                          # neural network activation function
    
    def _batcher(self, X, Y, batch_size=1):
        ''' Return a complete dataset but by parts of size "batch_size" '''
        p = X.shape[1]                                        # number of patterns
        li, ui = 0, batch_size                                # inferior and superior indexes 
        
        while True:
            if li < p:
                yield X[:, li:ui], Y[:, li:ui]
                li, ui = li+batch_size, ui+batch_size
            else:
                return None
    
    def predict(self, X):
        ''' Make predictions  '''
        Z = np.dot(self.w, X) + self.b
        return self.f(Z)
    
    def train(self, X, Y, epochs=200, lr=0.1, batch_size=1):
        ''' This function trains the neural network with gradient descent algorithm'''
        
        p = X.shape[1]                                      # number of patterns
        for _ in range(epochs):                             # for every epoch 
            miniBatch= self._batcher(X, Y, batch_size)      # minibatch for training quickly
            for mX, mY in miniBatch:                        # for every minibatch in dataset
                # propagation
                Z = np.dot(self.w, mX) + self.b
                # get the result of the activation functionand and its derivative
                Ypred, DY = self.f(Z, derivative=True)
                # local gradient
                lg = (mY-Ypred) * DY
                # adjust the synaptic weights and the bias vector
                self.w += (lr/p) * np.dot(lg, mX.T)
                self.b += (lr/p) * np.sum(lg, axis=1).reshape(-1,1)
            
            
            
            
            
        
        
