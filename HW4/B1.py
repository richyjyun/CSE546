import csv
import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

data = []

with open('ml-100k\\u.data') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t')
    for row in spamreader:
        data.append([int(row[0])-1, int(row[1])-1, int(row[2])])
data = np.array(data)

num_observations = len(data)   # num_observations = 100,000
num_users = max(data[:, 0])+1  # num_users = 943, indexed 0,...,942
num_items = max(data[:, 1])+1  # num_items = 1682 indexed 0,...,1681

np.random.seed(1)
num_train = int(0.8*num_observations)
perm = np.random.permutation(data.shape[0])
train = data[perm[0:num_train], :]
test = data[perm[num_train::], :]

def mse(a, b):
    return np.mean(np.square(a-b))

"""
a)
"""
# Calculate average
avgScore = np.zeros(num_items)
trainItems = train[:, 1]
trainItems = trainItems[np.unique(trainItems)]
for i in range(len(trainItems)):
    avgScore[trainItems[i]] = np.mean(train[train[:, 1] == trainItems[i], 2])

# Do prediction
predictions = avgScore[test[:, 1]]
error = mse(predictions, test[:, 2])

"""
b)
"""
# Set up R tilde
Rtrain = np.zeros((num_items, num_users))
for i in range(len(train)):
    Rtrain[train[i, 1], train[i, 0]] = train[i, 2]

# Do SVD and predictions
d = [1, 2, 5, 10, 20, 50]
trainerr = np.zeros(len(d))
testerr = np.zeros(len(d))
for i in range(len(d)):
    u, s, vt = svds(Rtrain, k=d[i])
    reconst = np.matmul(u*s, vt)
    trainpred = reconst[train[:, 1], train[:, 0]]
    trainerr[i] = mse(train[:, 2], trainpred)
    testpred = reconst[test[:, 1], test[:, 0]]
    testerr[i] = mse(test[:, 2], testpred)

# Plot
plt.plot(d, trainerr)
plt.plot(d, testerr)
plt.xlabel('Dimensions')
plt.ylabel('MSE')
plt.legend(('Train', 'Test'))

"""
c)
"""
def Loss(U, V, lam, t):
    a = 0
    for i in range(len(t)):
        temp = np.matmul(U[t[i, 1]], np.transpose(V[t[i, 0]]))
        temp = np.sum(np.square(temp - t[i, 2]))
        a = a + temp
    b = lam*np.sum(np.square(np.linalg.norm(U[t[:, 1]], 2, axis=1)))
    c = lam*np.sum(np.square(np.linalg.norm(V[t[:, 0]], 2, axis=1)))
    return a+b+c

d = 10
sigma = 2
lam = 1e-4
u = np.random.rand(num_items, d)*sigma
v = np.random.rand(num_users, d)*sigma