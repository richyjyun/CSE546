import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
import seaborn as sns

sns.set()

# Load data
mndata = MNIST('../MNIST/data/')
X_train, labels_train = map(np.array, mndata.load_training())
X_test, labels_test = map(np.array, mndata.load_testing())

X_train = X_train/255.0
X_test = X_test/255.0

# Define mu and sigma
mu = np.mean(X_train, axis=0)
Sig = np.matmul(np.transpose(X_train-mu), X_train-mu)/60000

# Eigenvalue decomposition. Need to flip as the order is wrong
D, U = np.linalg.eigh(Sig)
D = np.flip(D)
U = np.fliplr(U)

def reconst(x, m, u, k):
    temp = np.matmul(u[:, 0:k], np.transpose(u[:, 0:k]))
    return np.transpose(np.matmul(temp, np.transpose(x-m))) + m

def mse(a, b):
    temp = np.square(a-b)
    return np.mean(temp)

# Get reconstruction errors
train_err = []
test_err = []
for k in range(1, 101):
    print(k)
    pred = reconst(X_train, mu, U, k)
    train_err.append(mse(X_train, pred))
    pred = reconst(X_test, mu, U, k)
    test_err.append(mse(X_test, pred))

# a
temp = [1, 2, 10, 30, 50]
for i in temp:
    print(D[i-1])
print(np.sum(D))

# c
x = np.arange(1, 101, 1)
plt.plot(x, train_err)
plt.plot(x, test_err)
plt.xlabel('k')
plt.ylabel('Mean Squared Error')
plt.legend(('Train', 'Test'))

plt.figure()
valsum = np.zeros(100)
for i in range(100):
    valsum[i] = 1-(np.sum(D[0:i])/np.sum(D))
plt.plot(x, valsum)
plt.xlabel('k')
plt.ylabel('1 - ratio of eigenvalues')

# d
plt.figure()
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(np.reshape(U[:, i], (28, 28)))
    plt.title(i+1)
    plt.axis('off')

# e
plt.figure()
digs = [2, 6, 7]
ks = [5, 15, 40, 100]
for d in range(len(digs)):
    ind = np.nonzero(labels_train == digs[d])
    ind = np.squeeze(np.asarray(ind))
    xtemp = X_train[np.random.choice(ind), :]
    for k in range(len(ks)):
        pred = reconst(xtemp, mu, U, ks[k])
        pred = np.reshape(pred, (28, 28))
        plt.subplot(len(digs), len(ks), d*len(ks)+k+1)
        plt.imshow(pred)
        plt.axis('off')
        if d == 0:
            plt.title('k='+str(ks[k]))






