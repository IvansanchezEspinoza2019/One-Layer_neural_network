from neural_network import *
import numpy as np

### Example 
minx = -5
maxx = 5

classes = 4
p_c = 20
X = np.zeros((2, classes * p_c))
Y = np.zeros((classes, classes * p_c))

for i in range(classes):
    seed = minx + (maxx-minx) * np.random.rand(2,1)
    X[:, i*p_c:(i+1)*p_c] = seed + 0.15 * np.random.randn(2, p_c)
    Y[i, i*p_c:(i+1)*p_c] = np.ones((1, p_c))
    

net = OLN(2, classes, softmax)
net.train(X, Y, epochs=150, lr=1, batch_size=1)
Ypred = net.predict(X)

import matplotlib.pyplot as plt

cm  = [[1,1,0],[1,0,1],[0,1,1],[1,0,0],
       [0,1,0],[0,0,1],[1,1,1],[0,0,0]]

ax1=plt.subplot(1,2,1)
y_c = np.argmax(Y, axis=0)
for i in range(X.shape[1]):
    ax1.plot(X[0,i], X[1, i], '*',c=cm[y_c[i]])
ax1.axis([-5.5,5.5,-5.5,5.5])
ax1.set_title('Original Problem')
ax1.grid()

ax2=plt.subplot(1,2,2)
y_c = np.argmax(Ypred, axis=0)
for i in range(X.shape[1]):
    ax2.plot(X[0,i], X[1, i], '*',c=cm[y_c[i]])
ax2.axis([-5.5,5.5,-5.5,5.5])
ax2.set_title('Net Prediction')
ax2.grid()






