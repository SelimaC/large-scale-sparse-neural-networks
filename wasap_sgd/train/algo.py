### Algo class

from .optimizer import get_optimizer


class Algo(object):
    """The Algo class contains all information about the training algorithm """

    # available options and their default values
    supported_opts = {
                      'validate_every': 1000,
                      'sync_every': 1,
                      'mode': 'sgdm',
                      'optimizer_params': '{}'
    }

    def __init__(self, optimizer, **kwargs):
        """optimizer: string naming an optimization algorithm as defined in Optimizer.get_optimizer()
            Configuration options should be provided as keyword arguments.
            Available arguments are:
               loss: string naming the loss function to be used for training
               validate_every: number of time steps to wait between validations
               sync_every: number of time steps to wait before getting weights from parent
               mode: 'sgd' or 'sgdm' are supported
            Optimizer configuration options should be provided as additional
            named arguments (check your chosen optimizer class for details)."""
        for opt in self.supported_opts:
            if opt in kwargs:
                setattr(self, opt, kwargs[opt])
            else:
                setattr(self, opt, self.supported_opts[opt])

        self.optimizer_name = optimizer
        if optimizer is not None:
            optimizer_args = {arg: val for arg, val in kwargs.items()
                              if arg not in self.supported_opts}

            self.optimizer = get_optimizer(optimizer)(**optimizer_args)
        else:
            self.optimizer = None

        """ Workers are only responsible for computing the gradient and 
            sending it to the master, so we use ordinary SGD with learning rate 1 and 
            compute the gradient as (old weights - new weights) after each batch."""

        self.step_counter = 0
        self.worker_update_type = 'update'

    def get_config(self):
        config = {}
        config['optimizer'] = str(self.optimizer_name)
        for opt in self.supported_opts:
            config[opt] = str(getattr(self, opt))
        return config

    def __str__(self):
        strs = ["optimizer: " + str(self.optimizer_name)]
        strs += [opt + ": " + str(getattr(self, opt)) for opt in self.supported_opts]
        return '\n'.join(strs)

    ### For Worker ###
    def compute_update(self, cur_weights, new_weights):
        """Computes the update to be sent to the parent process"""
        if self.worker_update_type == 'weights':
            return new_weights
        else:
            update = {'pdw': {}, 'pdd': {}}
            update['pdw'] = new_weights['pdw']
            update['pdd'] = new_weights['pdd']
            return update

    def set_worker_model_weights(self, model, weights):
        """Apply a new set of weights to the worker's copy of the model"""
        model.set_weights(weights)

    def should_sync(self):
        """Determine whether to pull weights from the master"""
        self.step_counter += 1
        return self.step_counter % self.sync_every == 0

    ### For Master ###
    def apply_update(self, weights, update, epoch, sync=False, retain=False):
        """Calls the optimizer to apply an update
            and returns the resulting weights"""
        return self.optimizer.apply_update(weights, update, epoch, sync, retain)
