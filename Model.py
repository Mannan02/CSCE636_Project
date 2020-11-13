### YOUR CODE HERE
# import tensorflow as tf
import torch
import torch.nn as nn
import os, time
import numpy as np
from Network import MyNetwork, ResidualBlock
import tqdm
from ImageUtils import parse_record

"""This script defines the training, validation and testing process.
"""

class MyModel(object):
    num_epochs = 2
    learning_rate = 0.001
    def __init__(self, configs):
        self.configs = configs
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = MyNetwork(ResidualBlock, [2, 2, 2])

    def model_setup(self):
        print('---Setup input interfaces...')

    # For updating learning rate
    def update_lr(self,optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def train(self, x_train, y_train, configs, x_valid=None, y_valid=None):

        batch_size = 100
        self.network.train()
        curr_lr = self.learning_rate
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)  # initialize optimizer
        # criterion = nn.BCEWithLogitsLoss(reduction='mean')  # initialize loss function
        criterion = nn.CrossEntropyLoss()
        num_batches = int(np.ceil(x_train.shape[0] / batch_size))

        for _ in range(self.num_epochs):
            idxs = np.arange(x_train.shape[0])
            np.random.shuffle(idxs)

            qbar = tqdm.tqdm(range(num_batches))
            for i in qbar:
                idx = idxs[i: min(i + batch_size, x_train.shape[0])]
                X_batch = []
                for batch_i in idx:
                    X_batch.append(parse_record(x_train[batch_i], True))
                y_batch = y_train[idx]
                X_batch_tensor = torch.tensor(X_batch)
                X_batch_tensor = X_batch_tensor.transpose(1,3)
                y_batch_tensor = torch.tensor(y_batch).long()
                if torch.cuda.is_available():
                    X_batch_tensor = X_batch_tensor.cuda()
                    y_batch_tensor = y_batch_tensor.cuda()
                    self.network = self.network.cuda()

                y_pred = self.network(X_batch_tensor)
                loss = criterion(y_pred, y_batch_tensor)

                # back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (_ + 1) % 20 == 0:
                curr_lr /= 3
                self.update_lr(optimizer, curr_lr)
            # do validation
            if x_valid is not None and y_valid is not None:
                score = self.evaluate(x_valid, y_valid)
                print("score = {} in validation set.\n".format(score))

    def evaluate(self, X, y):
        preds = self.predict_prob(X)
        # is_correct = (y == preds).astype(np.float64)
        # score = np.sum(is_correct) / is_correct.size
        _, predicted = torch.max(preds.data, 1)
        total = y.size
        y = torch.Tensor(y)
        correct = (predicted == y).sum().item()
        return 100 * correct / total

    def predict_prob(self, X):
        X = [parse_record(X[i], True) for i in range(len(X))]
        X_tensor = torch.tensor(X)
        X_tensor = X_tensor.transpose(1, 3)
        if torch.cuda.is_available():
            X_tensor = X_tensor.cuda()
            self.network = self.network.cuda()

        self.network.eval()

        with torch.no_grad():

            preds = self.network(X_tensor)


        return preds


### END CODE HERE