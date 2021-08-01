### Optimizers used to update master process weights

import numpy as np
import logging


class Optimizer(object):
    """Base class for optimization algorithms.
        Currently doesn't do anything."""

    def __init__(self):
        pass

    def reset(self):
        pass

    def apply_update(self, weights, gradient):
        raise NotImplementedError


class VanillaSGD(Optimizer):
    """Stochastic gradient descent with no extra frills.
          learning_rate: learning rate parameter for SGD"""

    def __init__(self, lr):
        super(VanillaSGD, self).__init__()
        self.learning_rate = lr

    def apply_update(self, weights, gradient):
        """Move weights in the direction of the gradient, by the amount of the
            learning rate."""

        for index, v in gradient.items():
            dw = v[0]
            delta = v[1]

            dw = retain_valid_updates(weights['w'][index], dw)

            weights['pdw'][index] = - self.learning_rate * dw
            weights['pdd'][index] = - self.learning_rate * delta

            weights['w'][index] += weights['pdw'][index]
            weights['b'][index] += weights['pdd'][index]

        return weights


class MomentumSGD(Optimizer):
    """Stochastic gradient descent with momentum and weight decay
          learning_rate: learning rate parameter for SGD"""

    def __init__(self, lr, weight_decay, momentum, n_workers):
        super(MomentumSGD, self).__init__()
        self.learning_rate = lr
        self.base_lr = 0.01
        self.epoch = 0
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr_decay = 0.1
        self.milestones = [5, 100, 175]
        self.current_milestone = 0
        self.n_workers = n_workers

    def apply_update(self, weights, gradient, epoch=0, sync=False, retain=False, nesterov=False):
        """Move weights in the direction of the gradient, by the amount of the
            learning rate."""

        self.epoch = epoch

        if sync:
            # Leaning rate scheduler
            if self.epoch <= 5:  # Gradually warmup phase
                self.learning_rate = self.base_lr * ((self.n_workers - 1.0) * self.epoch / 5 + 1.0)

            if self.epoch >= 200:  # First decay
                self.learning_rate *= self.lr_decay

            if self.epoch >= 275:  # Second decay
                self.learning_rate *= self.lr_decay

        for index, v in gradient.items():
            dw = v[0]
            delta = v[1]

            if not sync and retain:
               dw = retain_valid_updates(weights['w'][index], dw)

            # perform the update with momentum
            if index not in weights['pdw']:
                weights['pdw'][index] = - self.learning_rate * dw
                weights['pdd'][index] = - self.learning_rate * delta
            else:
                weights['pdw'][index] = self.momentum * weights['pdw'][index] - self.learning_rate * dw
                weights['pdd'][index] = self.momentum * weights['pdd'][index] - self.learning_rate * delta

            if nesterov:
                weights['w'][index] += self.momentum * weights['pdw'][index] - self.learning_rate * dw - self.weight_decay * weights['w'][index]
                weights['b'][index] += self.momentum * weights['pdd'][index] - self.learning_rate * delta - self.weight_decay * weights['b'][index]
            else:
                weights['w'][index] += weights['pdw'][index] - self.weight_decay * weights['w'][index]
                weights['b'][index] += weights['pdd'][index] - self.weight_decay * weights['b'][index]

        return weights


def sparse_divide_nonzero(a, b):
    inv_b = b.copy()
    inv_b.data = 1 / (inv_b.data + 1e-16)
    return a.multiply(inv_b)


def get_optimizer(name):
    """Get optimizer class by string identifier"""
    lookup = {
            # Native optimizers
            'sgd':           VanillaSGD,
            'sgdm':          MomentumSGD
            }
    return lookup[name]


def array_intersect(A, B):
    # this are for array intersection
    nrows, ncols = A.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)], 'formats': ncols * [A.dtype]}
    return np.in1d(A.view(dtype), B.view(dtype), assume_unique=True)  # boolean return


def retain_valid_updates(weights, gradient):
    cols = gradient.shape[1]
    weights = weights.tocoo()
    gradient = gradient.tocoo()
    K_weights = np.array(weights.row * cols + weights.col)
    K_gradient = np.array(gradient.row * cols + gradient.col)

    indices = np.setdiff1d(K_gradient, K_weights, assume_unique=True)
    if len(indices) != 0:
        rows, cols = np.unravel_index(indices, gradient.shape)
        gradient = gradient.tocsr()
        gradient[rows, cols] = 0
        gradient.eliminate_zeros()

    return gradient


class OptimizerBuilder(object):
    """Builds a  optimizer"""

    def __init__(self, name, config=None):
        self.name = name
        self.config = config
        if self.config is None:
            self.config = {}
        if self.name == 'sgd' and 'lr' not in self.config:
            logging.warning("Learning rate for SGD not set, using 0.1")
            self.config['lr'] = 0.1

    def build(self):
        opt_config = {'class_name': self.name, 'config': self.config}
        return opt_config
