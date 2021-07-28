import logging
import os
import sys
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
from numba import njit, prange
from utils.nn_functions import MSE, CrossEntropy

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
sys.stderr = stderr


@njit(parallel=True, fastmath=True, cache=True)
def backpropagation_updates_numpy(a, delta, rows, cols, out):
    for i in prange(out.shape[0]):
        s = 0
        for j in range(a.shape[0]):
            s += a[j, rows[i]] * delta[j, cols[i]]
        out[i] = s / a.shape[0]


@njit(fastmath=True, cache=True)
def find_first_pos(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


@njit(fastmath=True, cache=True)
def find_last_pos(array, value):
    idx = (np.abs(array - value))[::-1].argmin()
    return array.shape[0] - idx


@njit(fastmath=True, cache=True)
def compute_accuracy(activations, y_test):
    correct_classification = 0
    for j in range(y_test.shape[0]):
        if np.argmax(activations[j]) == np.argmax(y_test[j]):
            correct_classification += 1
    return correct_classification / y_test.shape[0]


@njit(fastmath=True, cache=True)
def dropout(x, rate):
    noise_shape = x.shape
    noise = np.random.uniform(0., 1., noise_shape)
    keep_prob = 1. - rate
    scale = np.float32(1 / keep_prob)
    keep_mask = noise >= rate
    return x * scale * keep_mask, keep_mask


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


def create_sparse_weights_II(epsilon, noRows, noCols):
    # generate an Erdos Renyi sparse weights mask
    weights=lil_matrix((noRows, noCols))
    for i in range(epsilon * (noRows + noCols)):
        weights[np.random.randint(0,noRows),np.random.randint(0,noCols)]=np.float64(np.random.randn()/10)
    # print ("Create sparse matrix with ", weights.getnnz(), " connections and ",(weights.getnnz()/(noRows * noCols))*100, "% density level")
    weights=weights.tocsr()
    return weights


def create_sparse_weights(epsilon, n_rows, n_cols, weight_init):
    # He uniform initialization
    if weight_init == 'he_uniform':
        limit = np.sqrt(6. / float(n_rows))

    # Xavier initialization
    if weight_init == 'xavier':
        limit = np.sqrt(6. / (float(n_rows) + float(n_cols)))

    mask_weights = np.random.rand(n_rows, n_cols)
    prob = 1 - (epsilon * (n_rows + n_cols)) / (n_rows * n_cols)  # normal to have 8x connections

    # generate an Erdos Renyi sparse weights mask
    weights = lil_matrix((n_rows, n_cols))
    n_params = np.count_nonzero(mask_weights[mask_weights >= prob])
    weights[mask_weights >= prob] = np.random.uniform(-limit, limit, n_params)
    # print("Create sparse matrix with ", weights.getnnz(), " connections and ",
    #       (weights.getnnz() / (n_rows * n_cols)) * 100, "% density level")
    weights = weights.tocsr()
    return weights


def array_intersect(a, b):
    # this are for array intersection
    n_rows, n_cols = a.shape
    dtype = {'names': ['f{}'.format(i) for i in range(n_cols)], 'formats': n_cols * [a.dtype]}
    return np.in1d(a.view(dtype), b.view(dtype))  # boolean return


class SETMPIModel(object):
    """Class that abstract all details of the model
    """

    def __init__(self, dimensions, activations, class_weights, **config):
        """
        :param dimensions: (tpl/ list) Dimensions of the neural net. (input, hidden layer, output)
        :param activations: (tpl/ list) Activations functions.
        :config activations: (tpl/ list) Activations functions.
        """
        self.n_layers = len(dimensions)

        self.n_layers = len(dimensions)
        self.learning_rate = config['lr']
        self.momentum = config['momentum']
        self.epochs = config['n_epochs']
        self.weight_decay = config['weight_decay']
        self.epsilon = config['epsilon']  # control the sparsity level as discussed in the paper
        self.zeta = config['zeta']  # the fraction of the weights removed
        self.dropout_rate = config['dropout_rate']  # dropout rate
        self.dimensions = dimensions
        self.batch_size = config['batch_size']
        self.weight_init = config['weight_init']
        self.class_weights = class_weights
        self.prune = config['prune']
        self.base_lr = 0.01
        self.lr_decay = 0.5
        self.num_workers = config['num_workers']
        self.momentum_correction = 1.

        self.save_filename = ""
        self.input_layer_connections = []

        # Weights and biases are initiated by index. For a one hidden layer net you will have a w[1] and w[2]
        self.w = {}
        self.b = {}
        self.pdw = {}
        self.pdd = {}
        self.activations = {}

        # Weights and biases are initiated by index. For a one hidden layer net you will have a w[1] and w[2]
        self.w = {}
        self.b = {}
        self.pdw = {}
        self.pdd = {}
        self.activations = {}

        for i in range(len(dimensions) - 1):
            if self.weight_init == 'normal':
                self.w[i + 1] = create_sparse_weights_II(self.epsilon, dimensions[i],
                                                         dimensions[i + 1])  # create sparse weight matrices
            else:
                self.w[i + 1] = create_sparse_weights(self.epsilon, dimensions[i], dimensions[i + 1],
                                                      weight_init=self.weight_init)  # create sparse weight matrices
            self.b[i + 1] = np.zeros(dimensions[i + 1], dtype='float32')
            self.activations[i + 2] = activations[i]

        if config['loss'] == 'mse':
            self.loss = MSE(self.activations[self.n_layers])
        elif config['loss'] == 'cross_entropy':
            self.loss = CrossEntropy()
        else:
            raise NotImplementedError("The given loss function is  ot implemented")

    def print_metrics(self, metrics):
        names = ['loss', 'accuracy']
        for name, metric in zip(names, metrics):
            logging.info("{0}: {1:.3f}".format(name, metric))

    def format_update(self):
        return {}

    def get_weights(self):
        """
                Retrieve the network parameters.
                :return: model parameters.
        """

        params = {
            'w': self.w,
            'b': self.b,
            'pdw': self.pdw,
            'pdd': self.pdd,
        }

        return params

    def set_weights(self, params):
        self.w = params['w']
        self.b = params['b']
        self.pdw = params['pdw']
        self.pdd = params['pdd']

    def compute_loss(self, y, y_hat):
        return self.loss.loss(y, y_hat)

    def _feed_forward(self, x, drop=False):
        """
        Execute a forward feed through the network.
        :param x: (array) Batch of input data vectors.
        :return: (tpl) Node outputs and activations per layer. The numbering of the output is equivalent to the layer numbers.
        """
        # w(x) + b
        z = {}

        # activations: f(z)
        a = {1: x}  # First layer has no activations as input. The input x is the input.
        masks = {}

        for i in range(1, self.n_layers):
            z[i + 1] = a[i] @ self.w[i] + self.b[i]
            a[i + 1] = self.activations[i + 1].activation(z[i + 1])
            if drop:
                if i < self.n_layers - 1:
                    # apply dropout
                    a[i + 1], keep_mask = dropout(a[i + 1], self.dropout_rate)
                    masks[i + 1] = keep_mask

        return z, a, masks

    def _back_prop(self, z, a, masks, y_true):
        """
        The input dicts keys represent the layers of the net.
        a = { 1: x,
              2: f(w1(x) + b1)
              3: f(w2(a2) + b2)
              4: f(w3(a3) + b3)
              5: f(w4(a4) + b4)
              }
        :param z: (dict) w(x) + b
        :param a: (dict) f(z)
        :param y_true: (array) One hot encoded truth vector.
        :return:
        """
        keep_prob = 1.
        if self.dropout_rate > 0:
            keep_prob = np.float32(1. - self.dropout_rate)

        # Determine partial derivative and delta for the output layer.
        # delta output layer
        delta = self.loss.delta(y_true, a[self.n_layers])
        dw = coo_matrix(self.w[self.n_layers - 1], dtype='float32')
        # compute backpropagation updates
        backpropagation_updates_numpy(a[self.n_layers - 1], delta, dw.row, dw.col, dw.data)

        update_params = {
            self.n_layers - 1: (dw.tocsr(),  np.mean(delta, axis=0))
        }

        # In case of three layer net will iterate over i = 2 and i = 1
        # Determine partial derivative and delta for the rest of the layers.
        # Each iteration requires the delta from the previous layer, propagating backwards.
        for i in reversed(range(2, self.n_layers)):
            # dropout for the backpropagation step
            if keep_prob != 1:
                delta = (delta @ self.w[i].transpose()) * self.activations[i].prime(z[i])
                delta = delta * masks[i]
                delta /= keep_prob
            else:
                delta = (delta @ self.w[i].transpose()) * self.activations[i].prime(z[i])

            dw = coo_matrix(self.w[i - 1], dtype='float32')

            # compute backpropagation updates
            backpropagation_updates_numpy(a[i - 1], delta, dw.row, dw.col, dw.data)

            update_params[i - 1] = (dw.tocsr(),  np.mean(delta, axis=0))

        return update_params

    def apply_update(self, gradient, epoch=0, sync=False, retain=False, worker=False):
        """Move weights in the direction of the gradient, by the amount of the
            learning rate."""

        old_lr = self.learning_rate
        if sync and not worker:
            # Leaning rate scheduler
            if epoch <= 5:  # Gradually warmup phase
                self.learning_rate = self.base_lr * ((self.num_workers - 1.0) * epoch / 5 + 1.0)

        self.momentum_correction = self.learning_rate / old_lr

        for index, v in gradient.items():
            dw = v[0]
            delta = v[1]

            if not sync and retain:
               dw = retain_valid_updates(self.w[index], dw)

            # perform the update with momentum
            if not worker and sync:
                self._update_w_b(index, dw/float(self.num_workers), delta/float(self.num_workers), worker, self.learning_rate)
            else:
                self._update_w_b(index, dw, delta, worker, self.learning_rate)

    def _update_w_b(self, index, dw, delta, worker=False, learning_rate=0.01, nesterov=False):
        """
        Update weights and biases.
        :param index: (int) Number of the layer
        :param dw: (array) Partial derivatives
        :param delta: (array) Delta error.
        """

        # perform the update with momentum
        if index not in self.pdw:
            self.pdw[index] = - learning_rate * dw
            self.pdd[index] = - learning_rate * delta
        else:
            self.pdw[index] = self.momentum * self.momentum_correction * self.pdw[index] - learning_rate * dw
            self.pdd[index] = self.momentum * self.momentum_correction * self.pdd[index] - learning_rate * delta

        if nesterov and not worker:
            self.w[index] += self.momentum * self.pdw[index] - learning_rate * dw - self.weight_decay * self.w[index]
            self.b[index] += self.momentum * self.pdd[index] - learning_rate * delta - self.weight_decay * self.b[index]
        else:
            self.w[index] += self.pdw[index] - self.weight_decay * self.w[index]
            self.b[index] += self.pdd[index] - self.weight_decay * self.b[index]

    def train_on_batch(self, x, y):
        z, a, masks = self._feed_forward(x, True)
        return self._back_prop(z, a, masks, y)

    def test_on_batch(self, x, y):
        accuracy, activations = self.predict(x, y)
        return self.loss.loss(y, activations), accuracy

    def weight_evolution(self, epoch, worker=False):
        # this represents the core of the SET procedure. It removes the weights closest to zero in each layer and add new random weights
        for i in range(1, self.n_layers - 1):
            # uncomment line below to stop evolution of dense weights more than 80% non-zeros
            # if self.w[i].count_nonzero() / (self.w[i].get_shape()[0]*self.w[i].get_shape()[1]) < 0.8:

            if self.prune and not worker and (epoch % 20 == 0 and epoch > 200):
                sum_incoming_weights = np.abs(self.w[i]).sum(axis=0)
                t = np.percentile(sum_incoming_weights, 10)
                sum_incoming_weights = np.where(sum_incoming_weights <= t, 0, sum_incoming_weights)
                ids = np.argwhere(sum_incoming_weights == 0)

                weights = self.w[i].tolil()
                pdw = self.pdw[i].tolil()
                weights[:, ids[:, 1]] = 0
                pdw[:, ids[:, 1]] = 0

                self.w[i] = weights.tocsr()
                self.pdw[i] = pdw.tocsr()

            # converting to COO form - Added by Amar
            wcoo = self.w[i].tocoo()
            vals_w = wcoo.data
            rows_w = wcoo.row
            cols_w = wcoo.col

            pdcoo = self.pdw[i].tocoo()
            vals_pd = pdcoo.data
            rows_pd = pdcoo.row
            cols_pd = pdcoo.col
            # print("Number of non zeros in W and PD matrix before evolution in layer",i,[np.size(valsW), np.size(valsPD)])
            values = np.sort(self.w[i].data)
            first_zero_pos = find_first_pos(values, 0)
            last_zero_pos = find_last_pos(values, 0)

            largest_negative = values[int((1 - self.zeta) * first_zero_pos)]
            smallest_positive = values[
                int(min(values.shape[0] - 1, last_zero_pos + self.zeta * (values.shape[0] - last_zero_pos)))]

            # remove the weights (W) closest to zero and modify PD as well
            vals_w_new = vals_w[(vals_w > smallest_positive) | (vals_w < largest_negative)]
            rows_w_new = rows_w[(vals_w > smallest_positive) | (vals_w < largest_negative)]
            cols_w_new = cols_w[(vals_w > smallest_positive) | (vals_w < largest_negative)]

            new_w_row_col_index = np.stack((rows_w_new, cols_w_new), axis=-1)
            old_pd_row_col_index = np.stack((rows_pd, cols_pd), axis=-1)

            new_pd_row_col_index_flag = array_intersect(old_pd_row_col_index,
                                                        new_w_row_col_index)  # careful about order

            vals_pd_new = vals_pd[new_pd_row_col_index_flag]
            rows_pd_new = rows_pd[new_pd_row_col_index_flag]
            cols_pd_new = cols_pd[new_pd_row_col_index_flag]

            self.pdw[i] = coo_matrix((vals_pd_new, (rows_pd_new, cols_pd_new)),
                                     (self.dimensions[i - 1], self.dimensions[i])).tocsr()

            # add new random connections
            keep_connections = np.size(rows_w_new)
            length_random = vals_w.shape[0] - keep_connections

            if self.weight_init == 'he_uniform':
                limit = np.sqrt(6. / float(self.dimensions[i - 1]))
            if self.weight_init == 'xavier':
                limit = np.sqrt(6. / (float(self.dimensions[i - 1]) + float(self.dimensions[i])))

            random_vals = np.random.uniform(-limit, limit, length_random)
            zero_vals = 0 * random_vals  # explicit zeros

            # adding  (wdok[ik,jk]!=0): condition
            while length_random > 0:
                ik = np.random.randint(0, self.dimensions[i - 1], size=length_random, dtype='int32')
                jk = np.random.randint(0, self.dimensions[i], size=length_random, dtype='int32')

                random_w_row_col_index = np.stack((ik, jk), axis=-1)
                random_w_row_col_index = np.unique(random_w_row_col_index,
                                                   axis=0)  # removing duplicates in new rows&cols
                oldW_row_col_index = np.stack((rows_w_new, cols_w_new), axis=-1)

                unique_flag = ~array_intersect(random_w_row_col_index,
                                               oldW_row_col_index)  # careful about order & tilda

                ik_new = random_w_row_col_index[unique_flag][:, 0]
                jk_new = random_w_row_col_index[unique_flag][:, 1]
                # be careful - row size and col size needs to be verified
                rows_w_new = np.append(rows_w_new, ik_new)
                cols_w_new = np.append(cols_w_new, jk_new)

                length_random = vals_w.shape[0] - np.size(rows_w_new)  # this will constantly reduce lengthRandom

            # adding all the values along with corresponding row and column indices - Added by Amar
            vals_w_new = np.append(vals_w_new, random_vals)
            if vals_w_new.shape[0] != rows_w_new.shape[0]:
                print("not good")
            self.w[i] = coo_matrix((vals_w_new, (rows_w_new, cols_w_new)),
                                   (self.dimensions[i - 1], self.dimensions[i])).tocsr()

    def predict(self, x_test, y_test, batch_size=100):
        """
        :param x_test: (array) Test input
        :param y_test: (array) Correct test output
        :param batch_size:
        :return: (flt) Classification accuracy
        :return: (array) A 2D array of shape (n_cases, n_classes).
        """
        activations = np.zeros((y_test.shape[0], y_test.shape[1]))
        for j in range(x_test.shape[0] // batch_size):
            k = j * batch_size
            l = (j + 1) * batch_size
            _, a_test, _ = self._feed_forward(x_test[k:l], drop=False)
            activations[k:l] = a_test[self.n_layers]
        accuracy = compute_accuracy(activations, y_test)
        return accuracy, activations
