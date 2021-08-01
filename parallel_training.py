import argparse
import logging
import tensorflow as tf
from mpi4py import MPI
from time import time
from utils.load_data import *
from utils.nn_functions import AlternatedLeftReLU, Softmax, Relu, Sigmoid
from wasap_sgd.mpi.manager import MPIManager
from wasap_sgd.train.algo import Algo
from wasap_sgd.train.data import Data
from wasap_sgd.train.model import SETMPIModel
from wasap_sgd.logger import initialize_logger

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Run this file with "mpiexec -n 6 python parallel_training.py"
# Add --synchronous if you want to train in synchronous mode
# Add --monitor to enable cpu and memory monitoring

# Uncomment next lines for debugging with size > 1 (note that port mapping ids change at very run)
# size = MPI.COMM_WORLD.Get_size()
# rank = MPI.COMM_WORLD.Get_rank()
# import pydevd_pycharm
# port_mapping = [56131, 56135] # Add ids of processes you want to debug in this list
# pydevd_pycharm.settrace('localhost', port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)


def shared_partitions(n, num_workers, batch_size):
    """"
    Split the training dataset equally amongst the workers
    """
    dinds = list(range(n))
    num_batches = n // batch_size
    worker_size = num_batches // num_workers

    data = dict.fromkeys(list(range(num_workers)))

    for w in range(num_workers):
        data[w] = dinds[w * batch_size * worker_size: (w+1) * batch_size * worker_size]

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--monitor', help='Monitor cpu and gpu utilization', default=False, action='store_true')

    # Configuration of network topology
    parser.add_argument('--masters', help='number of master processes', default=1, type=int)
    parser.add_argument('--processes', help='number of processes per worker', default=1, type=int)
    parser.add_argument('--synchronous', help='run in synchronous mode', action='store_true')

    # Configuration of training process
    parser.add_argument('--loss', help='loss function', default='cross_entropy')
    parser.add_argument('--sync-every', help='how often to sync weights with master',
                        default=1, type=int, dest='sync_every')
    parser.add_argument('--mode', help='Mode of operation.'
                        'One of "sgd" (Stohastic Gradient Descent), "sgdm" (Stohastic Gradient Descent with Momentum)',
                        default='sgdm')

    # logging configuration
    parser.add_argument('--log-file', default=None, dest='log_file',
                        help='log file to write, in additon to output stream')
    parser.add_argument('--log-level', default='info', dest='log_level', help='log level (debug, info, warn, error)')

    # Model configuration
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10,  help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--lr-rate-decay', type=float, default=0.0, help='learning rate decay (default: 0)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--dropout-rate', type=float, default=0.3, help='Dropout rate (default: 0.3)')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay (l2 regularization)')
    parser.add_argument('--epsilon', type=int, default=20, help='Sparsity level (default: 20)')
    parser.add_argument('--zeta', type=float, default=0.3,
                        help='It gives the percentage of unimportant connections which are removed and replaced with '
                             'random ones after every epoch(in [0..1])')
    parser.add_argument('--n-neurons', type=int, default=3000, help='Number of neurons in the hidden layer')
    parser.add_argument('--prune', default=True, help='Perform Importance Pruning', action='store_true')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--n-training-samples', type=int, default=60000, help='Number of training samples')
    parser.add_argument('--n-testing-samples', type=int, default=10000, help='Number of testing samples')
    parser.add_argument('--augmentation', default=True, help='Data augmentation', action='store_true')
    parser.add_argument('--dataset', default='fashionmnist', help='Specify dataset. One of "cifar10", "fashionmnist",'
                                                             '"madelon",  or "mnist"')

    args = parser.parse_args()

    # Default weight initialization technique
    weight_init = 'xavier'
    prune = args.prune
    n_hidden_neurons = args.n_neurons
    epsilon = args.epsilon
    zeta = args.zeta
    n_epochs = args.epochs
    batch_size = args.batch_size
    dropout_rate = args.dropout_rate
    learning_rate = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay
    n_training_samples = args.n_training_samples
    n_testing_samples = args.n_testing_samples
    learning_rate_decay = args.lr_rate_decay
    class_weights = None

    # Comment this if you would like to use the full power of randomization. I use it to have repeatable results.
    np.random.seed(args.seed)

    # Model architecture
    if args.dataset == 'fashionmnist' or args.dataset == 'mnist':
        # Model architecture mnist
        dimensions = (784, 1000, 1000, 1000, 10)
        loss = 'cross_entropy'
        weight_init = 'he_uniform'
        activations = (AlternatedLeftReLU(-0.6), AlternatedLeftReLU(0.6), AlternatedLeftReLU(-0.6), Softmax)
        if args.dataset == 'fashionmnist':
            X_train, Y_train, X_test, Y_test = load_fashion_mnist_data(args.n_training_samples, args.n_testing_samples)
        else:
            X_train, Y_train, X_test, Y_test = load_mnist_data(args.n_training_samples, args.n_testing_samples)
    elif args.dataset == 'madalon':
        # Model architecture madalon
        dimensions = (500, 400, 100, 400, 1)
        loss = 'mse'
        activations = (Relu, Relu, Relu, Sigmoid)
        X_train, Y_train, X_test, Y_test = load_madelon_data()
    elif args.dataset == 'cifar10':
        # Model architecture cifar10
        dimensions = (3072, 4000, 1000, 4000, 10)
        weight_init = 'he_uniform'
        loss = 'cross_entropy'
        activations = (AlternatedLeftReLU(-0.75), AlternatedLeftReLU(0.75), AlternatedLeftReLU(-0.75), Softmax)
        if args.augmentation:
            X_train, Y_train, X_test, Y_test = load_cifar10_data_not_flattened(args.n_training_samples, args.n_testing_samples)
        else:
            X_train, Y_train, X_test, Y_test = load_cifar10_data(args.n_training_samples, args.n_testing_samples)
    else:
        raise NotImplementedError("The given dataset is not available")

    comm = MPI.COMM_WORLD.Dup()

    model_weights = None
    rank = comm.Get_rank()
    num_processes = comm.Get_size()
    num_workers = num_processes - 1

    # Scale up the learning rate for synchronous training
    if args.synchronous:
        learning_rate = learning_rate * (num_workers)

    # Initialize logger
    base_file_name = "results/set_mlp_parallel_" + str(args.dataset)+"_" + str(args.epochs) + "_epochs_e" + \
                     str(epsilon) + "_rand" + str(args.seed) + "_num_workers_" + str(num_workers)
    log_file = base_file_name + "_logs_execution.txt"

    save_filename = base_file_name + "_process_" + str(rank)

    initialize_logger(filename=log_file, file_level=args.log_level, stream_level=args.log_level)

    if num_processes == 1:
        validate_every = int(X_train.shape[0] // (batch_size * args.sync_every))
        data = Data(batch_size=batch_size,
                    x_train=X_train, y_train=Y_train,
                    x_test=X_test, y_test=Y_test, augmentation=True,
                    dataset=args.dataset)
    else:
        if rank != 0:
            validate_every = int(X_train.shape[0] // (batch_size * args.sync_every))
            partitions = shared_partitions(X_train.shape[0], num_workers, batch_size)
            data = Data(batch_size=batch_size,
                        x_train=X_train[partitions[rank - 1]], y_train=Y_train[partitions[rank - 1]],
                        x_test=X_test, y_test=Y_test, augmentation=args.augmentation,
                        dataset=args.dataset)
            logging.info(f"Data partition contains {data.x_train.shape[0]} samples")
        else:
            validate_every = int(X_train.shape[0] // batch_size)
            if args.synchronous:
                validate_every = int(X_train.shape[0] // (batch_size * num_workers))
            data = Data(batch_size=batch_size,
                        x_train=X_train, y_train=Y_train,
                        x_test=X_test, y_test=Y_test, augmentation=args.augmentation,
                        dataset=args.dataset)
            logging.info(f"Validate every {validate_every} time steps")
    del X_train, Y_train, X_test, Y_test

    # SET parameters
    model_config = {
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'dropout_rate': dropout_rate,
        'lr': learning_rate,
        'zeta': zeta,
        'epsilon': epsilon,
        'momentum': momentum,
        'weight_decay': 0.0,
        'n_hidden_neurons': n_hidden_neurons,
        'n_training_samples': n_training_samples,
        'n_testing_samples': n_testing_samples,
        'loss': loss,
        'weight_init': weight_init,
        'prune': prune,
        'num_workers': num_workers
    }

    # Some input arguments may be ignored depending on chosen algorithm
    if args.mode == 'sgdm':
        algo = Algo(optimizer='sgdm', validate_every=validate_every, lr=learning_rate,
                    sync_every=args.sync_every, weight_decay=args.weight_decay, momentum=args.momentum, n_workers=num_workers)
    elif args.mode == 'sgd':
        algo = Algo(optimizer='sgd', validate_every=validate_every, lr=learning_rate, sync_every=args.sync_every)
    else:
        raise NotImplementedError("The given optimizer is not available")

    start_time = time()
    # Instantiate SET model
    model = SETMPIModel(dimensions, activations, class_weights=class_weights, master=(rank == 0), **model_config)

    step_time = time() - start_time
    if rank == 0:
        logging.info(f"Model creation time:  {step_time}")

    # Creating the MPIManager object causes all needed worker and master nodes to be created
    manager = MPIManager(comm=comm, data=data, algo=algo, model=model,
                         num_epochs=args.epochs, num_masters=args.masters,
                         num_processes=args.processes, synchronous=args.synchronous,
                         monitor=args.monitor, save_filename=save_filename)

    # Process 0 launches the training procedure
    if rank == 0:
        logging.debug('Training configuration: %s', algo.get_config())

        t_0 = time()
        histories = manager.process.train()
        delta_t = time() - t_0
        manager.free_comms()
        logging.info("Total execution time is {0:.3f} seconds".format(delta_t))
        logging.info("Testing time is {0:.3f} seconds".format(manager.process.validate_time))
        delta_t -= manager.process.validate_time
        logging.info("Training finished in {0:.3f} seconds".format(delta_t))
        logging.info("Evolution time is {0:.3f} seconds".format(manager.process.evolution_time))

        logging.info("------------------------------------------------------------------------------------------------")
        logging.info("Final performance of the model on the test dataset")
        manager.process.validate()

    comm.barrier()
    logging.info("Terminating")
