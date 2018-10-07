import torch as T
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm, trange
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from myTorch import *

T.set_default_tensor_type('torch.FloatTensor')

batch_size = 32
nb_epochs  = 5000
nb_digits  = 10

train_loader = T.utils.data.DataLoader(datasets.MNIST(
    './data', train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True
) 
test_loader = T.utils.data.DataLoader(datasets.MNIST(
    './data', train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True
) 

x_train = train_loader.dataset.train_data.view(-1, 784).float()
x_test  = test_loader.dataset.test_data.view(-1,   784).float()
y_train = train_loader.dataset.train_labels.view(-1, 1).float()
y_test  = test_loader.dataset.test_labels.view(-1,   1).float()

def one_hot(y):
    y_onehot = T.FloatTensor(len(y), 10) 
    y_onehot.zero_()
    return y_onehot.scatter_(1, y.long().view(-1, 1), 1)

y_train = one_hot(y_train)
y_test  = one_hot(y_test)

#because hinge require [-1, 1]
y_train = y_train * 2 - 1
y_test  = y_test  * 2 - 1

loss = Hinge()
model = Linear(784, 10)

epsilon = 1e-9
train_error, test_error = [], []

for i in trange(nb_epochs):
    #make a random batch
    idx = [np.random.randint(0, x_train.shape[0]) for i in range(batch_size)]
    x = x_train[idx].view(-1, 784)
    y = y_train[idx].view(-1,  10)
    #train
    model.zero_grad()
    yhat = model(x)
    err  = loss(y, yhat)
    delta = loss.backward(y, yhat)
    model.backward_update_gradient(x, delta)
    model.update_parameters(epsilon)
    train_error.append(err.sum() / 10)
    #test
    yhat = model(x_test)
    err  = loss.forward(y_test, yhat)
    test_error.append(err.sum() / 10)


plt.figure(figsize=(10, 5))
plt.suptitle("Hinge MNIST")
plt.subplot(1, 2, 1)
plt.plot(np.arange(nb_epochs), train_error)
plt.title("Train")
plt.subplot(1, 2, 2)
plt.title("Test")
plt.plot(np.arange(nb_epochs), test_error)
plt.legend()
plt.show()




