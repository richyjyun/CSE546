import numpy as np
import scipy.spatial.distance as sp
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def fstar(x):
    return 4*np.sin(np.pi*x)*np.cos(6*np.pi*np.square(x))

def kpoly(x1, x2, d):
    temp = np.matmul(x1, np.transpose(x2))
    return np.power(1 + temp, d)

def krbf(x1, x2, g):
    dist = sp.cdist(x1, x2, 'euclidean')
    return np.exp(-g*np.square(dist))

def err(f, y):
    return np.mean(np.square(f - y))

def alpha(k, l, y):
    return np.matmul(np.linalg.inv(k+l*np.eye(len(k))), y)

n = 300
cv = 10  # cross validation width
x = np.random.uniform(0, 1, (n, 1))
e = np.random.normal(0, 1, (n, 1))
y = fstar(x) + e

# Set range of hyperparameters
hp = np.linspace(1, 50, 50)  # for rbf
#hp = np.linspace(10, 40, 50)  # for poly

# Set range of lambda
lambdas = np.logspace(-8, -1, 50)

errors = np.zeros((len(lambdas), len(hp)))

# Loop through all parameters
for l in range(len(lambdas)):
    lam = lambdas[l]
    for h in range(len(hp)):
        print(l, h)
        errs = np.zeros(n)
        allK = krbf(x, x, hp[h])

        # Cross validation
        for i in range(int(n/cv)):
            inds = np.arange((i-1)*cv, i*cv, 1)
            tempx = np.delete(x, inds, axis=0)
            tempy = np.delete(y, inds, axis=0)
            K = krbf(tempx, tempx, hp[h])
            a = alpha(K, lam, tempy)

            K = krbf(np.expand_dims(x[inds, 0], axis=1), tempx, i)
            f = np.dot(K, a)

            errs[i] = np.sum(np.square(f-y[inds]))

        # Calculate total error
        errors[l, h] = np.sum(errs)/n

# Find optimal hyperparameters
temp = np.where(errors == np.min(errors))
optlam = lambdas[temp[0][0]]
opthp = hp[temp[1][0]]

# Find optimal alpha
K = krbf(x, x, opthp)
a = alpha(K, optlam, y)

# Get f and fhat
plotx = np.arange(0, 1, 0.01)
plotx = np.expand_dims(plotx, axis=1)
plotk = krbf(plotx, x, opthp)
fhat = np.matmul(plotk, a)
f = fstar(plotx)

# Plot
2plt.scatter(x, y, 10, 'k')
plt.plot(plotx, f)
plt.plot(plotx, fhat)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(('f_true', 'f_poly', 'f_rbf', 'Data'))

plt.show()

plt.imshow(errors, origin='lower')
plt.colorbar()
plt.xlabel('hyperparameter')
plt.ylabel('lambda')
plt.show()