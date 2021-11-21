import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import seaborn as sns
import scipy.io as sp

sns.set()
# Load data
data = sp.loadmat('C:/Users/Richy Yun/Dropbox/Fetz Lab/EvokedSpikes/NNdata.mat')
X = data['PrevSpike']
Y = data['ESmat']

perm = torch.randperm(len(X))
X_train = X[perm[0:2400], :]
X_test = X[perm[2401:len(perm)], :]
Y_train = Y[perm[0:2400], :]
Y_test = Y[perm[2401:len(perm)], :]

# Rearrange data
X_train = torch.from_numpy(X_train)
X_train = X_train.type(torch.FloatTensor)
X_test = torch.from_numpy(X_test)
X_test = X_test.type(torch.FloatTensor)

Y_train = torch.from_numpy(Y_train)
Y_train = Y_train.type(torch.FloatTensor)
Y_test = torch.from_numpy(Y_test)
Y_test = Y_test.type(torch.FloatTensor)

# Initialize variables
d = X_train.size()[1]
train_samples = len(Y_train)
k = 10
ha = 64
hb = 32
alpha = 1/np.sqrt(d)

# for a
W0a = torch.rand(d, ha)*2*alpha-alpha
W0a = W0a.requires_grad_()
B0a = torch.rand(1, ha)*2*alpha-alpha
B0a = B0a.requires_grad_()
W1a = torch.rand(ha, k)*2*alpha-alpha
W1a = W1a.requires_grad_()
B1a = torch.rand(1, k)*2*alpha-alpha
B1a = B1a.requires_grad_()
# for b
W0b = torch.rand(d, hb)*2*alpha-alpha
W0b = W0b.requires_grad_()
B0b = torch.rand(1, hb)*2*alpha-alpha
B0b = B0b.requires_grad_()
W1b = torch.rand(hb, hb)*2*alpha-alpha
W1b = W1b.requires_grad_()
B1b = torch.rand(1, hb)*2*alpha-alpha
B1b = B1b.requires_grad_()
W2b = torch.rand(hb, k)*2*alpha-alpha
W2b = W2b.requires_grad_()
B2b = torch.rand(1, k)*2*alpha-alpha
B2b = B2b.requires_grad_()


# a. Wide shallow network
class F1(torch.nn.Module):
    def __init__(self, w0, b0, w1, b1):
        super().__init__()
        self.w0 = torch.nn.Parameter(w0)
        self.b0 = torch.nn.Parameter(b0)
        self.w1 = torch.nn.Parameter(w1)
        self.b1 = torch.nn.Parameter(b1)

    def forward(self, x):
        temp = F.relu(torch.matmul(x, self.w0) + self.b0)
        return torch.matmul(temp, self.w1) + self.b1


# b. Narrow deep network
class F2(torch.nn.Module):
    def __init__(self, w0, b0, w1, b1, w2, b2):
        super().__init__()
        self.w0 = torch.nn.Parameter(w0)
        self.b0 = torch.nn.Parameter(b0)
        self.w1 = torch.nn.Parameter(w1)
        self.b1 = torch.nn.Parameter(b1)
        self.w2 = torch.nn.Parameter(w2)
        self.b2 = torch.nn.Parameter(b2)

    def forward(self, x):
        temp = F.relu(torch.matmul(x, self.w0) + self.b0)
        temp = F.relu(torch.matmul(temp, self.w1) + self.b1)
        return torch.matmul(temp, self.w2) + self.b2


# Set training parameters
lr = 1e-3
n_epochs = 2000
batch = 1000
loops = int(train_samples/batch)
#model = F1(W0a, B0a, W1a, B1a)  # choose between F1 and F2 here
model = F2(W0b, B0b, W1b, B1b, W2b, B2b)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Train
train_acc = []
losses = []
acc = 0
count = 0
loss = 1
while loss > 0.1:
    # for debugging
    print(count)
    count = count+1

    # mini-batch steps
    perm = torch.randperm(train_samples)
    for b in range(loops):
        ind = perm[b*batch:((b+1)*batch-1)]
        y_hat = model(X_train[ind, :])

        # cross entropy combines softmax calculation with NLLLoss
        loss = F.mse_loss(y_hat, Y_train[ind, :])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Calculate accuracy and loss per epoch
    y_hat = model(X_train)
    # y = y_hat.argmax(axis=1)
    # temp = sum(y == Y_train)
    # acc = temp.item()/len(Y_train)
    # train_acc.append(acc)
    loss = F.mse_loss(y_hat, Y_train)
    losses.append(loss)
    print(loss)  # for debugging

# Plot accuracy and loss
plt.figure()
x = range(1, len(losses)+1)
plt.subplot(1, 2, 1)
plt.plot(x, [i * 100 for i in train_acc])
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.subplot(1, 2, 2)
plt.plot(x, losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')