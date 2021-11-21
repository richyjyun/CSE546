import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
import random
import seaborn as sns

sns.set()

def rearrangeData(d, l):
    ind = np.logical_or(l == 2, l == 7)
    data = d[ind, :]
    label = l[ind]
    label = label.astype('int16')
    label[label == 2] = -1
    label[label == 7] = 1
    return data, label

def muwb(x, y, w, b):
    return 1/(1+np.exp(-y * (b + np.matmul(x, w))))

def objective(x, y, w, b, n, lam):
    return np.sum(np.log(1/muwb(x, y, w, b)))/n + lam*np.linalg.norm(w, 2)**2

# Load data
mndata = MNIST('../MNIST/data/')
X_train, labels_train = map(np.array, mndata.load_training())
X_test, labels_test = map(np.array, mndata.load_testing())

# Rearrange data
X_train = X_train/255.0
X_test = X_test/255.0
X_train, labels_train = rearrangeData(X_train, labels_train)
X_test, labels_test = rearrangeData(X_test, labels_test)

# Initialize variables
n = np.shape(X_train)[0]
ntest = np.shape(X_test)[0]
d = np.shape(X_train)[1]
w = np.zeros(d)
b = 0
lam = 0.1
eta = 0.1
trainerr = []
testerr = []
obj = []
objtest = []
lasterr = 100
batch = 10

# rescale lambda
lam = lam / (n/batch)

while lasterr > 3:

    # random sampling
    ind = random.sample(range(n), batch)
    x = X_train[ind, :]
    y = labels_train[ind]

    oldw = w.copy()
    gradw = -np.matmul(y * (1-muwb(x, y, oldw, b)), x) / batch + 2*lam*w
    w = w - eta*(gradw)
    gradb = -np.matmul(y, (1-muwb(x, y, oldw, b))) / batch
    b = b - eta*(gradb)

    obj.append(objective(X_train, labels_train, w, b, n, lam))
    objtest.append(objective(X_test, labels_test, w, b, ntest, lam))

    err = np.sign(np.matmul(X_train, w)+b) # probability y=1 (that it is 7)
    trainerr.append((1-np.count_nonzero(err == labels_train)/n)*100)
    err = np.sign(np.matmul(X_test, w)+b)
    testerr.append((1-np.count_nonzero(err == labels_test)/ntest)*100)

    lasterr = trainerr[len(trainerr)-1]
    print(lasterr)

# plot i
plt.plot(obj)
plt.plot(objtest)
plt.xlabel('Iteration')
plt.ylabel('Regularized Negative Log Objective')
plt.legend(('Train', 'Test'))

# plot ii
plt.figure()
plt.plot(trainerr)
plt.plot(testerr)
plt.xlabel('Iteration')
plt.ylabel('Error (%)')
plt.legend(('Train', 'Test'))