import torch
import torch as T
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split

from myTorch import *

torch.set_default_tensor_type('torch.FloatTensor')



data = pd.read_csv("data/housing.data", delimiter="\\s+", header=None)
x_data = data.as_matrix()[:, :-1]
y_data = data.as_matrix()[:,  -1]


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=.8)
x_train, x_test = T.tensor(x_train).float(), T.tensor(x_test).float()
y_train, y_test = T.tensor(y_train).float(), T.tensor(y_test).float()


epsilon = 1e-10
train_error, test_error = [], []

loss = MSE()
model = Linear(13, 1)

batch_size = 16
nb_epochs  = 5000
for i in trange(nb_epochs):
    #make a random batch
    idx = [np.random.randint(0, x_train.shape[0]) for i in range(batch_size)]
    x = x_train[idx].view(-1, 13)
    y = y_train[idx].view(-1,  1)
    #train
    model.zero_grad()
    yhat = model(x)
    err = loss(y, yhat)
    delta = loss.backward(y, yhat)
    model.backward_update_gradient(x, delta)
    model.update_parameters(epsilon)
    train_error.append(err / len(x_train))
    #test
    err = loss(y_test, model.forward(x_test))
    test_error.append(err / len(x_test))


plt.figure(figsize=(10, 5))
plt.suptitle("MiniBatch MSE")
plt.subplot(1, 2, 1)
plt.plot(np.arange(nb_epochs), train_error)
plt.title("Train")
plt.subplot(1, 2, 2)
plt.title("Test")
plt.plot(np.arange(nb_epochs), test_error)
plt.legend()
plt.show()



