#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import time
import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels command: python hw1-q1.py perceptron -epochs 100
        """
        y_hat = self.predict(X)         
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        predicted_label = self.predict(x_i) 
        if predicted_label != y_i:
            self.W[y_i] +=  x_i
            self.W[predicted_label] -= x_i
        return
    
class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001, l2_penalty=0.0, **kwargs):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        scores = np.dot(self.W, x_i)                        # (n_classes)
        exp_scores = np.exp(scores - np.max(scores))        # (n_classes)
        probabilities = exp_scores / np.sum(exp_scores)     # (n_classes)
        one_hot_y = np.zeros(self.W.shape[0])               # (n_classes)
        one_hot_y[y_i] = 1

        gradients = np.outer((probabilities - one_hot_y), x_i)
        gradients += l2_penalty * self.W

        self.W -= learning_rate * gradients
        
class MLP(object):
    def __init__(self, n_classes, n_features, hidden_size=100):
        # Initialize an MLP with a single hidden layer.

        self.W_1 = np.random.normal(loc=0.1, scale=0.1, size=(hidden_size, n_features)) 
        self.b_1 = np.zeros((hidden_size,1))
        self.W_2 = np.random.normal(loc=0.1, scale=0.1, size=(n_classes, hidden_size))
        self.b_2 = np.zeros((n_classes,1))

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes.
        z = np.dot(self.W_1, np.transpose(X)) + self.b_1            # size (hidden_size, n_examles), each column is an example, each row is a node
        h = np.maximum(0, z)                                        # ReLu function of z, size(hidden_size, n_examples), eac
        scores = np.dot(self.W_2, h) + self.b_2                     # weights to output layer, size(n_classes, n_examples), each row is one node
        exp_scores = np.exp(scores - np.max(scores, axis=0))        # for stability, (n_classes, n_examples)
        probabilities = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)     # softmax, size (n_classes, n_examples)
        predicted_labels = probabilities.argmax(axis=0)             # pick the largest prob. size (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    """
    def train_epoch(self, X, y, learning_rate=0.001, **kwargs):
        """
        #Dont forget to return the loss of the epoch.
    """     
        # shuffle the input
        #indexes = np.arange(len(X))
        #np.random.shuffle(indexes)
        #X_shuffled = X[indexes]
        #y_shuffled = y[indexes]
        start_time = time.time()
        # set up mini_batches, default=1
        batch_size = 1
        num_batches = len(X) // batch_size
        loss = 0
        #print('main for loop')

        # do num_batches amount of gradient updates
        for i in range(num_batches):
            X_batch = X[i * batch_size : (i + 1) * batch_size]
            y_batch = y[i * batch_size : (i + 1) * batch_size]

        # predict
            #print('predict)')
            z = np.dot(self.W_1, np.transpose(X_batch)) + self.b_1                      # size (hidden_size, n_examles), each column is an example, each row is a node
            h = np.maximum(0, z)                                                        # ReLu function of z, size(hidden_size, n_examples), eac
            scores = np.dot(self.W_2, h) + self.b_2                                     # weights to output layer, size(n_classes, n_examples), each row is one node
            exp_scores = np.exp(scores - np.max(scores, axis=0))                        # for stability, (n_classes, n_examples)
            probabilities = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)      # softmax, size (n_classes, n_examples)

            # create one hot encoded matrix_like probabilities
            one_hot = np.zeros_like(probabilities)                                      # (n_classes, n_examples)
            one_hot[y_batch, np.arange(len(y_batch))] = 1

            # compute negative log-likelihood loss function
            eps = 1e-99                                                                 # for numerical stability
            loss += np.sum(-np.log(probabilities[one_hot == 1] + eps))

            # compute gradients
            #print('compute gradients')
            gradient_z2 = probabilities - one_hot                                       # dL/dz2, size(n_classes, n_examples) 

            #gradient_b2 = np.sum(gradient_z2, axis=1, keepdims=True) / batch_size       # dL/db2 (n_classes, 1)
            gradient_W2 = np.dot(gradient_z2, np.transpose(h)) / batch_size             # dL/dW2 (n_classes, hidden_size)
            
            #gradient_h1 = np.dot(np.transpose(self.W_2), gradient_z2)                   # dL/dh1 (hidden_size, n_examples)
            gradient_z2 = np.dot(np.transpose(self.W_2), gradient_z2)                   # dL/dh1 (hidden_size, n_examples)

            self.b_2 -= learning_rate * (np.sum(gradient_z2, axis=1, keepdims=True) / batch_size)       # dL/db2 (n_classes, 1)
            self.W_2 -= learning_rate * (np.dot(gradient_z2, np.transpose(h)) / batch_size)             # dL/dW2 (n_classes, hidden_size)

            gradient_relu = np.zeros_like(gradient_z2)                                  # dh1/dz (hidden_size, n_examples)
            gradient_relu[h > 0] = 1
            #gradient_z1 = gradient_z2 * gradient_relu                                   # dL/dz1 = dL/dh1 * dh1//dz1 (hidden_size, n_examples)
            gradient_z1 = gradient_z2 * gradient_relu                                   # dL/dz1 = dL/dh1 * dh1//dz1 (hidden_size, n_examples)
            
            gradient_b1 = np.sum(gradient_z1, axis=1, keepdims=True) / batch_size       # dL/db1 (hidden_size,1)
            gradient_W1 = np.dot(gradient_z1, X_batch) / batch_size                     # dL/dW1 (hidden_size, n_features)
            
            # update weights
            self.W_2 -= learning_rate * gradient_W2
            self.b_2 -= learning_rate * gradient_b2

            self.W_1 -= learning_rate * gradient_W1
            self.b_1 -= learning_rate * gradient_b1
        endtime = time.time()
        elapsed_time = endtime - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

        return loss / (num_batches * batch_size)                                        # normalize by n_samples
        """

    def train_epoch(self, X, y, learning_rate=0.001, **kwargs):
        """
        Dont forget to return the loss of the epoch.
        """     
        loss = 0
        n_samples = X.shape[0]

        # do n_samles amount of gradient updates
        for idx in range(n_samples):
            y_i = y[idx]
            x_i = X[idx].reshape(-1, 1)

        # predict
            z = np.dot(self.W_1, x_i) + self.b_1                                        # size (hidden_size, n_examles)
            h = np.maximum(0, z)                                                        # ReLu function of z, size(hidden_size, n_examples)
            scores = np.dot(self.W_2, h) + self.b_2                                     # weights to output layer, size(n_classes, n_examples)
            exp_scores = np.exp(scores - np.max(scores, axis=0, keepdims=True))         # for stability, (n_classes, n_examples)
            probabilities = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)      # softmax, size (n_classes, n_examples)

            # compute negative log-likelihood loss function
            loss += np.sum(-np.log(probabilities[y_i]))

            # compute gradients of output layer
            gradients_l2 = probabilities
            gradients_l2[y_i] -= 1                                                      # dL/dz2, size(n_classes, 1) 

            # compute gradients of input layer
            gradients_l1 = np.dot(np.transpose(self.W_2), gradients_l2)                 # dL/dh1 (hidden_size, 1)
            gradients_l1[z <= 0] = 0                                                    # dL/dz1 = dL/dh1 * dh1//dz1 (hidden_size, 1)

            # update weights and biases
            self.b_2 -= learning_rate * gradients_l2                                    # dL/db2 = dL/dz2
            self.W_2 -= learning_rate * np.dot(gradients_l2, np.transpose(h))           # dL/dW2 (n_classes, hidden_size)

            self.b_1 -= learning_rate * gradients_l1                                    # dL/db1 = dL/dz1 (hidden_size, 1)
            self.W_1 -= learning_rate * np.dot(gradients_l1, np.transpose(x_i))         # dL/dW1 (hidden_size, n_features)
        return loss / n_samples                                                         # normalize by n_samples


def plot(epochs, train_accs, val_accs, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_loss(epochs, loss, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_w_norm(epochs, w_norms, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('W Norm')
    plt.plot(epochs, w_norms, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=100,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    parser.add_argument('-l2_penalty', type=float, default=0.0,)
    parser.add_argument('-data_path', type=str, default='intel_landscapes.npz',)
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_dataset(data_path=opt.data_path, bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    weight_norms = []
    valid_accs = []
    train_accs = []

    start = time.time()

    print('initial train acc: {:.4f} | initial val acc: {:.4f}'.format(
        model.evaluate(train_X, train_y), model.evaluate(dev_X, dev_y)
    ))
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate,
                l2_penalty=opt.l2_penalty,
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        elif opt.model == "logistic_regression":
            weight_norm = np.linalg.norm(model.W)
            print('train acc: {:.4f} | val acc: {:.4f} | W norm: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1], weight_norm,
            ))
            weight_norms.append(weight_norm)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs, filename=f"Q1-{opt.model}-accs.pdf")
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss, filename=f"Q1-{opt.model}-loss.pdf")
    elif opt.model == 'logistic_regression':
        plot_w_norm(epochs, weight_norms, filename=f"Q1-{opt.model}-w_norms.pdf")
    with open(f"Q1-{opt.model}-results.txt", "w") as f:
        f.write(f"Final test acc: {model.evaluate(test_X, test_y)}\n")
        f.write(f"Training time: {minutes} minutes and {seconds} seconds\n")


if __name__ == '__main__':
    main()
