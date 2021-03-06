#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from matplotlib import pyplot

class CNN(object):

    def __init__(self, conv1, pool1, conv2, pool2, neural, error):
        self.conv1 = conv1
        self.pool1 = pool1
        self.conv2 = conv2
        self.pool2 = pool2
        self.neural = neural
        self.error = error


    def train(self, X, T, epsilon, lam, gamma, s_batch, epochs):
        n_data = X.shape[0]
        self.__set_loss(epochs)
        for epo in range(epochs):
            perm = np.random.permutation(n_data)
            for i in range(0, n_data, s_batch):
                x, t = X[perm[i:i+s_batch]], T[perm[i:i+s_batch]].T

                # forward
                cv1, pl1, cv2, pl2, u, z, y = self.__forward(x, s_batch)

                # backward
                output_delta, hidden_delta, input_delta = self.neural.backward(t, y, z, u, self.error)
                pl2_delta, cv2_delta, pl1_delta, cv1_delta = self.__backward(input_delta, pl2, cv2, pl1, cv1)

                # update weight
                self.neural.update_weight(output_delta, hidden_delta, z, u, epsilon, lam)
                self.conv2.update_weight(cv2_delta, epsilon)
                self.conv1.update_weight(cv1_delta, epsilon)

                # accumulate loss
                self.__accumulate_loss(y, t, n_data, epo)

            # update learning rate
            if (epo + 1) % 10 == 0: epsilon = self.__update_epsilon(epsilon, gamma)

            print 'epoch: {0}, loss: {1}'.format(epo, self.__loss[epo])


    def predict(self, X):
        return self.__forward(X, X.shape[0])[6]


    def accuracy(self, Y, T):
        return (Y.argmax(axis = 0) == T.argmax(axis = 1)).sum() * 1.0 / Y.shape[1]


    def save_lossfig(self, fn = 'loss.png'):
        pyplot.plot(np.arange(self.__loss.size), self.__loss)
        pyplot.savefig(fn)


    def __forward(self, X, s_batch):
        cv1 = self.conv1.forward(X)
        pl1 = self.pool1.forward(cv1)
        cv2 = self.conv2.forward(pl1)
        pl2 = self.pool2.forward(cv2)
        u = pl2.reshape(s_batch, -1).T
        z, y = self.neural.forward(u)
        return cv1, pl1, cv2, pl2, u, z, y


    def __backward(self, input_delta, pl2, cv2, pl1, cv1):
        pl2_delta = input_delta.reshape(pl2.shape)
        cv2_delta = self.pool2.backward(cv2, pl2_delta, self.conv2.activator)
        pl1_delta = self.conv2.backward(cv2_delta, pl1.shape)
        cv1_delta = self.pool1.backward(cv1, pl1_delta, self.conv1.activator)
        return pl2_delta, cv2_delta, pl1_delta, cv1_delta


    def __update_epsilon(self, epsilon, gamma):
        return gamma * epsilon


    def __set_loss(self, epochs):
        self.__loss = np.zeros(epochs)


    def __accumulate_loss(self, y, t, n_data, epo):
        self.__loss[epo] += self.error.delta(y, t) / n_data

