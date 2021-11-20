import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
from scipy import linalg

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

# Load data
mndata = MNIST('./data/')
X_train, labels_train = map(np.array, mndata.load_training())
X_test, labels_test = map(np.array, mndata.load_testing())

# Rearrange data
X_train = X_train/255.0
X_test = X_test/255.0
Y_train = np.zeros((len(X_train), 10))
Y_train[range(0, len(X_train)), labels_train] = 1

# Train model
lam = 0.0001
W = train(X_train, Y_train, lam)

# Get predictions
P_train = predict(W, X_train)
P_test = predict(W, X_test)

# Error calculation
train_error = (len(X_train) - np.sum(P_train == labels_train))/len(X_train)
test_error = (len(X_test) - np.sum(P_test == labels_test))/len(X_test)

# Print output
print('Training error: ' + str(train_error*100) + '%')
print('Testing error: ' + str(test_error*100) + '%')