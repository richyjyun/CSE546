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
newY_train = Y_train[inds[0:samples], :]

mu = 0
var = 0.1
unilim = 2*np.pi
lam = 1e-4
d = int(np.size(newX_train)/ samples)
p = 6000

G = np.random.normal(mu, np.sqrt(var), (p, d))
b = np.random.uniform(0, unilim, (p, 1))

# Apply function and train model
h = hx(newX_train, G, b)
W = train(h, newY_train, lam)

# Apply function to test set and validate
h = hx(X_test, G, b)
P_test = predict(W, h)

# Calculate error
test_error = (len(P_test) - np.sum(P_test == labels_test)) / len(P_test)

delta = 0.05
a = 0
b = 1
m = len(X_test)

conf_int = np.sqrt((b-a)**2*np.log(2/delta)/(2*m))

print(test_error)
print(conf_int)