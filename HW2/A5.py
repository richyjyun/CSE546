import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set()

# to find the maximum lambda that gives all zero in w
def lambdamax(x, y, n):
    yres = y - np.sum(y) / n
    xres = np.matmul(x, yres)
    xsum = 2 * np.absolute(xres)
    return np.max(xsum)

df_train = pd.read_table("crime-train.txt")
df_test = pd.read_table("crime-test.txt")

temp = df_train.values
n = len(df_train)
d = 95
y = temp[:, 0]
x = np.transpose(temp[:, 1:d+1])
temp = df_test.values
ytest = temp[:, 0]
xtest = np.transpose(temp[:, 1:d+1])
lam = lambdamax(x, y, n)
w = np.zeros(d)

print('hi')

deltalim = 0.01
minlam = 0.1
nonzero = 0
lambdas = []
nonzeros = []
regpath = []
trainmse = []
testmse = []
oldlam = lam

# Keep checking smaller lambdas until at least 900 elements in w are nonzero
# a can be calculated outside the for loop
a = 2 * np.sum(np.square(x), axis=1)
while oldlam > minlam:
    lambdas.append(lam)
    print('lambda', lam)
    oldlam = lam

    # gradient descent
    maxdelta = float('inf')
    while maxdelta > deltalim:

        b = np.sum(y - np.matmul(w, x)) / n

        oldw = w.copy()

        for j in range(d):

            # calculate ck
            tempw = np.delete(w, j)
            tempx = np.delete(x, j, 0)
            ck = 2 * np.dot(np.transpose(x[j, :]), y - (b + np.matmul(tempw, tempx)))

            # determine wk
            if ck < -lam:
                w[j] = (ck + lam) / a[j]
            elif ck > lam:
                w[j] = (ck - lam) / a[j]
            else:
                w[j] = 0

        # find difference, set variables for conditions
        delta = np.absolute(oldw - w)
        maxdelta = np.max(delta)

        # sanity check to make sure objective gets smaller each iteration
        tempb = np.sum(y - np.matmul(w, x)) / n
        err = np.sum(np.square(np.matmul(w, x)+tempb-y)) + lam*np.sum(np.absolute(w))
        print('error: ', err)

    # save number of nonzero elements and set new lambda
    nonzero = np.count_nonzero(w)
    print('nonzero:', nonzero)
    nonzeros.append(nonzero)
    lam = lam / 2
    regpath.append(w.copy())
    error = np.sum(np.square(np.matmul(w, x)-y)) / n
    trainmse.append(error)
    error = np.sum(np.square(np.matmul(w, xtest)-ytest)) / np.shape(xtest)[1]
    testmse.append(error)

# plot a
plt.plot(lambdas, nonzeros)
plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('Non-zeros')

# plot b ind = 4, 13, 8, 6, 2
agePct12t29 = []
pctWSocSec = []
pctUrban = []
agePct65up = []
householdsize = []
for i in range(len(regpath)):
    agePct12t29.append(regpath[i][3])
    pctWSocSec.append(regpath[i][12])
    pctUrban.append(regpath[i][7])
    agePct65up.append(regpath[i][5])
    householdsize.append(regpath[i][1])
plt.plot(lambdas, agePct12t29)
plt.plot(lambdas, pctWSocSec)
plt.plot(lambdas, pctUrban)
plt.plot(lambdas, agePct65up)
plt.plot(lambdas, householdsize)
plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('Regularization Path')
plt.legend(('agePct12t29', 'pctWSocSec', 'pctUrban', 'agePct65up', 'householdsize'))

# plot c
plt.figure()
plt.plot(lambdas, trainmse)
plt.plot(lambdas, testmse)
plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('Mean squared-error')
plt.legend(('Train', 'Test'))

# plot d
plt.figure()
plt.plot(w)
MAX = df_train.columns[np.argmax(w)+1]
MIN = df_train.columns[np.argmin(w)+1]



