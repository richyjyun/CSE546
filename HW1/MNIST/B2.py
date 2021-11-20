import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mnist import MNIST
from scipy import linalg

sns.set()

def train(X, Y, lam):
    d = int(X.size/len(X))
    reg_matrix = lam * np.eye(d)
    a = np.matmul(X.transpose(), X) + reg_matrix
    b = np.matmul(X.transpose(), Y)
    W = linalg.solve(a, b)
    return W

def predict(W, X):
    d = len(X)
    Y = np.zeros(d)
    temp = np.matmul(X, W)
    for i in range(0, d):
        ind = np.where(temp[i, :] == np.amax(temp[i, :]))
        Y[i] = ind[0]
    return Y

def hx(X, G, b):
    return np.cos(np.matmul(X, np.transpose(G)) + np.transpose(b))

# Load data
mndata = MNIST('./data/')
X_train, labels_train = map(np.array, mndata.load_training())
X_test, labels_test = map(np.array, mndata.load_testing())

# Rearrange data
X_train = X_train/255.0
X_test = X_test/255.0
Y_train = np.zeros((len(X_train), 10))
Y_train[range(0, len(X_train)), labels_train] = 1

# Split randomly into train and validation
inds = np.random.permutation(len(X_train))
samples = int(len(X_train)*0.8)
newX_train = X_train[inds[0:samples], :]
newX_test = X_train[inds[samples:len(inds)], :]
newY_train = Y_train[inds[0:samples], :]
newlabels_train = labels_train[inds[0:samples]]
newlabels_test = labels_train[inds[samples:len(inds)]]

# Cross-validation
mu = 0
var = 0.1
unilim = 2*np.pi
lam = 1e-4
d = int(np.size(newX_train)/ samples)
P = np.arange(50, 6000, 50)
train_error = np.zeros(len(P))
test_error = np.zeros(len(P))

for i in range(0, len(P)):
    p = P[i]
    print(str(p))
    G = np.random.normal(mu, np.sqrt(var), (p, d))
    b = np.random.uniform(0, unilim, (p, 1))

    # Apply function and train model
    h = hx(newX_train, G, b)
    W = train(h, newY_train, lam)

    # Training set output
    P_train = predict(W, h)

    # Apply function to validation and validate
    h = hx(newX_test, G, b)
    P_test = predict(W, h)

    # Calculate error
    train_error[i] = (len(P_train) - np.sum(P_train == newlabels_train))/len(P_train)
    test_error[i] = (len(P_test) - np.sum(P_test == newlabels_test)) / len(P_test)

# Plot
plt.plot(P, train_error*100)
plt.plot(P, test_error*100)
plt.xlabel('p')
plt.ylabel('% Error')
plt.legend(['Training Error', 'Validation Error'])
plt.show()
