import numpy as np
import scipy.spatial.distance as sp
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

def fstar(x):
    return 4 * np.sin(np.pi * x) * np.cos(6 * np.pi * np.square(x))

def kpoly(x1, x2, d):
    temp = np.matmul(x1, np.transpose(x2))
    return np.power(1 + temp, d)

def krbf(x1, x2, g):
    dist = sp.cdist(x1, x2, 'euclidean')
    return np.exp(-g * np.square(dist))

def err(f, y):
    return np.mean(np.square(f - y))

def alpha(k, l, y):
    return np.matmul(np.linalg.inv(k + l * np.eye(len(k))), y)

# initialize variables
n = 300
lam = 5e-3
beta = 30
gamma = 25
boot = 300
x = np.random.uniform(0, 1, (n, 1))
e = np.random.normal(0, 1, (n, 1))
y = fstar(x) + e
finex = np.arange(0, 1, 0.01)
finex = np.expand_dims(finex, axis=1)
f = fstar(finex)

# bootstrap
CI_poly = np.zeros((len(finex), boot))
CI_rbf = np.zeros((len(finex), boot))
errors = np.zeros(boot)
for b in range(boot):
    ind = np.random.choice(range(n), n)
    newx = x[ind]
    newy = y[ind]
    K = kpoly(newx, newx, beta)
    apoly = alpha(K, lam, newy)
    K = krbf(newx, newx, gamma)
    arbf = alpha(K, lam, newy)
    plotk = kpoly(finex, newx, beta)
    CI_poly[:, b] = np.squeeze(np.matmul(plotk, apoly))
    plotk = krbf(finex, newx, gamma)
    CI_rbf[:, b] = np.squeeze(np.matmul(plotk, arbf))
    errors[b] = np.mean(np.square(f-CI_poly[:, b])-np.square(f-CI_rbf[:, b]))

# optimal curves
K = kpoly(x, x, beta)
apoly = alpha(K, lam, y)
K = krbf(x, x, gamma)
arbf = alpha(K, lam, y)

# plot variables
plotk = kpoly(finex, x, beta)
fhatpoly = np.matmul(plotk, apoly)
plotk = krbf(finex, x, gamma)
fhatrbf = np.matmul(plotk, arbf)

# plot
fig1 = plt.figure()
plt.scatter(x, y, 10, 'k')
plt.plot(finex, f, 'k')
plt.plot(finex, fhatpoly, 'b')
plt.plot(finex, fhatrbf, 'g')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(('f_true', 'f_poly', 'f_rbf', 'Data'))

# plot with confidence intervals
fig2 = plt.figure()
scatter1 = plt.scatter(x, y, 10, 'k', label='Data')
plt.plot(finex, f, 'k', label='f_true')
plt.plot(finex, fhatpoly, 'b', label='f_poly')
perc_poly = np.percentile(CI_poly, [5, 95], axis=1)
plt.plot(finex, perc_poly[0, :], 'b--', label='CI_poly')
plt.plot(finex, perc_poly[1, :], 'b--')
plt.plot(finex, fhatrbf, 'g', label='f_rbf')
perc_rbf = np.percentile(CI_rbf, [5, 95], axis=1)
plt.plot(finex, perc_rbf[0, :], 'g--', label='CI_rbf')
plt.plot(finex, perc_rbf[1, :], 'g--')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

perc_err = np.percentile(errors, [5, 95])

