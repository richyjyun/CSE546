import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as sp
import scipy.stats as st
from mnist import MNIST
import seaborn as sns
sns.set()

# kmeans++ initialization
def kmeansplusplus(data, k):
    n = np.size(data, axis=0)
    ind = np.random.choice(range(n))
    centroid = np.expand_dims(data[ind, :], axis=0)
    for c in range(k - 1):
        d = sp.cdist(data, centroid, 'euclidean') ** 2
        d = np.min(d, axis=1)
        d = d / np.sum(d)
        ind = np.random.choice(range(n), 1, p=d)
        centroid = np.append(centroid, data[ind, :], axis=0)
    return centroid

def kmeans(data, k):
    centroid = kmeansplusplus(data, k)
    maxchange = 100
    while maxchange > 0.001:
        d = sp.cdist(data, centroid, 'euclidean')
        clusters = np.argmin(d, 1)
        newcentroid = np.copy(centroid)
        for c in range(k):
            newcentroid[c, :] = np.mean(data[clusters == c, :], axis=0)
        maxchange = np.max(np.mean(np.abs(newcentroid - centroid), axis=1))
        centroid = np.copy(newcentroid)
        print(maxchange)

    d = sp.cdist(data, centroid, 'euclidean')
    clusters = np.argmin(d, 1)

    return centroid, clusters

# Load data
mndata = MNIST('../MNIST/data/')
X_train, labels_train = map(np.array, mndata.load_training())
X_test, labels_test = map(np.array, mndata.load_testing())

# Rearrange data
X_train = X_train/255.0
X_test = X_test/255.0

# Plot centroids
k = 10
cent, clust = kmeans(X_train, k)
for i in range(k):
    plt.subplot(2, k/2, i+1)
    plt.imshow(np.reshape(cent[i, :], (28, 28)))
    plt.axis('off')

# Train and test error as a function of k
k = [2, 4, 8, 16, 32, 64]
trainerr = np.zeros(len(k))
testerr = np.zeros(len(k))
for i in range(len(k)):
    cent, clust = kmeans(X_train, k[i])
    d = sp.cdist(X_train, cent, 'euclidean')
    clusters = np.argmin(d, 1)
    trainerr[i] = np.mean(np.square(np.sum(X_train - cent[clusters, :], axis=1)))
    d = sp.cdist(X_test, cent, 'euclidean')
    clusters = np.argmin(d, 1)
    testerr[i] = np.mean(np.square(np.sum(X_test - cent[clusters, :], axis=1)))

plt.plot(k, trainerr)
plt.plot(k, testerr)
plt.xlabel('k')
plt.ylabel('error')
plt.legend(('Train Error', 'Test Error'))
