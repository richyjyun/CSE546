import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
import torch
import seaborn as sns

sns.set()
# Load data
mndata = MNIST('../MNIST/data/')
X_train, labels_train = map(np.array, mndata.load_training())
X_test, labels_test = map(np.array, mndata.load_testing())

# Rearrange data
X_train = X_train/255.0
X_train = torch.from_numpy(X_train)
X_train = X_train.type(torch.FloatTensor)
X_test = X_test/255.0
X_test = torch.from_numpy(X_test)
X_test = X_test.type(torch.FloatTensor)

# for NLLLoss+softmax
Y_train = torch.from_numpy(labels_train)
Y_train = Y_train.type(torch.long)
Y_test = torch.from_numpy(labels_test)
Y_test = Y_test.type(torch.long)

## for MSE
# Y_train = np.zeros((len(X_train), 10))
# Y_train[range(0, len(X_train)), labels_train] = 1
# Y_train = torch.from_numpy(Y_train)
# Y_train = Y_train.type(torch.FloatTensor)
# Y_test = np.zeros((len(X_test), 10))
# Y_test[range(0, len(X_test)), labels_test] = 1
# Y_test = torch.from_numpy(Y_test)
# Y_test = Y_test.type(torch.FloatTensor)

W = torch.zeros(784, 10, requires_grad=True)
step_size = 0.1
max_step = 1
trainacc = []
testacc = []
for i in range(100):
    y_hat = torch.matmul(X_train, W)
    # cross entropy combines softmax calculation with NLLLoss

    loss = torch.nn.functional.cross_entropy(y_hat, Y_train) # for NLLLoss+softmax
    #loss = torch.nn.functional.mse_loss(y_hat, Y_train) # for MSE
    # computes derivatives of the loss with respect to W
    loss.backward()

    # gradient descent update
    W.data = W.data - step_size * W.grad
    max_step = W.grad.max().item()
    print(max_step)

    # .backward() accumulates gradients into W.grad instead
    # of overwriting, so we need to zero out the weights
    W.grad.zero_()

    # Save accuracy for each iteration
    y = torch.matmul(X_train, W).argmax(axis=1)
    y = sum(y == Y_train) # for NLLLoss
    #y = sum(y == Y_train.argmax(axis=1)) # for MSE
    trainacc.append(y.item()/X_train.size()[0]*100)
    print(y.item()/X_train.size()[0]*100)

    y = torch.matmul(X_test, W).argmax(axis=1)
    y = sum(y == Y_test) # for NLLLoss
    #y = sum(y == Y_test.argmax(axis=1)) # for MSE
    testacc.append(y.item()/X_test.size()[0]*100)

# Plot
plt.plot(trainacc)
plt.plot(testacc)
plt.ylabel('Accuracy (%)')
plt.xlabel('Iteration')
plt.title('Negative log-likelihood + Softmax')
plt.legend(('Train', 'Test'))
