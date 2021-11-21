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

Y_train = np.zeros((len(X_train), 10))
Y_train[range(0, len(X_train)), labels_train] = 1
Y_train = torch.from_numpy(Y_train)
Y_train = Y_train.type(torch.FloatTensor)
Y_test = np.zeros((len(X_test), 10))
Y_test[range(0, len(X_test)), labels_test] = 1
Y_test = torch.from_numpy(Y_test)
Y_test = Y_test.type(torch.FloatTensor)

# Initialize variables
d = 784
lr = 1e-3
epochs = 50
batch = 1000
loops = int(len(X_train)/batch)
H = [32, 64, 128]
TrainLoss = np.zeros((len(H), epochs))
TestLoss = np.zeros((len(H), epochs))

plotind = np.zeros(10)
for digit in range(10):         # indices of example digits, plot originals
    ind = np.nonzero(labels_train == digit)
    ind = np.squeeze(np.asarray(ind))
    plotind[digit] = np.random.choice(ind)
    x = torch.reshape(X_train[int(plotind[digit]), :], (28, 28))
    plt.subplot(len(H) + 1, 10, digit + 1)
    plt.imshow(x)
    plt.axis('off')

# Train autoencoder
for h in range(len(H)):
    # Define model
    model = torch.nn.Sequential(
        torch.nn.Linear(d, H[h]),
        torch.nn.ReLU(),
        torch.nn.Linear(H[h], d),
        torch.nn.ReLU()
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    count = 0
    acc = 0
    #while acc < 0.9:
    for e in range(epochs):
        print(e)
        # mini-batch
        perm = torch.randperm(len(X_train))
        for b in range(loops):
            ind = perm[b * batch:((b + 1) * batch)]
            tempx = X_train[ind, :]
            x_hat = model(tempx)

            # cross entropy combines softmax calculation with NLLLoss
            loss = torch.nn.functional.mse_loss(x_hat, tempx)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Save losses
        x_hat = model(X_train)
        loss = torch.nn.functional.mse_loss(x_hat, X_train)
        TrainLoss[h, e] = loss.item()
        print(loss.item())

        x_hat = model(X_test)
        loss = torch.nn.functional.mse_loss(x_hat, X_test)
        TestLoss[h, e] = loss.item()

    # Plot reconstruction
    x_hat = model(X_train)
    for digit in range(10):
        x = torch.reshape(x_hat[int(plotind[digit]), :], (28, 28))
        plt.subplot(len(H)+1, 10, digit + 1 + 10*(h+1))
        plt.imshow(x.data.numpy())
        plt.axis('off')

# Plot losses
plt.figure()
for i in range(len(H)):
    plt.plot(TrainLoss[i, :])
plt.ylabel('Training Loss')
plt.xlabel('Epochs')
plt.legend(('h=32', 'h=64', 'h=128'))

plt.figure()
plt.plot(TestLoss[0, :])
plt.ylabel('Test Loss')
plt.xlabel('Epochs')
plt.legend(('F1', 'F2'))