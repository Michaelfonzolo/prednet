import numpy as np
import os
import random

from keras import backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import Input, Dense, Flatten, TimeDistributed
from keras.models import Model

from prednet import PredNet
from data_utils import SequenceGenerator

from sms_notifier import *

# Requirements
# - Open a PrednetClient object
#   - With this object you can download data, choose whether you want to preload a trained
#     PredNet model or build a new one, analyze the training metrics, and output the predictions
#     for given input data.
#
# Desired Behaviour
#
#   >>> data_downloader = FTPDataLoader(output="Drone")
#   >>> data_downloader.connect('host')
#   >>> data_downloader.login('usr', 'pswrd', 'acct')
#   >>> data_downloader.download(hickle_dump="drone.hkl")
#   >>> X = data_downloader.get_data(sources='2011_09_20')
#   >>> 
#   >>> client = PredNetClient()
#   >>> client.load_pretrained()
#   >>> client.predict(X[:10], verbose=1)
#
# This would then give a readout of the MSE over time of the prediction, and a plot of the predicted
# frames against the original frames.

class PredNetClient(object):

    """
    A class which bundles the model parameters of a PredNetClient object.
    """
    class ModelParameters(object):
        
        def __init__(self,
                     input_spatial_dimensions=(128, 160), 
                     input_channels=3,
                     number_of_layers=4,
                     A_filter_sizes=(3,3,3),
                     A_hat_filter_sizes=(3,3,3,3),
                             R_filter_sizes=(3,3,3,3),
                             number_of_filters_per_layer=(3, 48, 96, 192),
                             loss_weights_per_layer=[1.0,.0,.0,.0],
                             number_of_timesteps=10,
                             channels_first=False,
                             **kwargs):
            self.input_spatial_dimensions   = input_spatial_dimensions
            self.input_channels             = input_channels
            self.number_of_layers           = number_of_layers

            self.A_filter_sizes     = A_filter_sizes
            self.A_hat_filter_sizes = A_hat_filter_sizes
            self.R_filter_sizes     = R_filter_sizes

            self.number_of_filters_per_layer = number_of_filters_per_layer
            self.loss_weights_per_layer      = np.array(loss_weights_per_layer)
            self.number_of_timesteps         = number_of_timesteps
            self.loss_weights_per_timestep   = kwargs.get("loss_weights_per_timestep",  
                np.array([0] + [1.0/(self.number_of_timesteps - 1) for i in range(self.number_of_timesteps - 1)]))

            self.channels_first = channels_first if channels_first is not None else K.image_data_format()

    """
    A class which bundles the training parameters of a PredNetClient object.
    """
    class TrainingParameters(object):
        
        def __init__(self,
                     epochs=150,
                     batch_size=4,
                     epoch_size=500,
                     number_of_validation_sequences=100,
                     optimizer='adam',
                     learning_rate_scheduler='default',
                     max_training_samples=30000,
                     **kwargs):

            # Only (training_file and training_source) or (training_files) can be specified, but
            # not both simultaneously.
            if "training_files" in kwargs:
                # Note: "training_sources" isn't actually an accepted key word parameter, but we guard against it regardless.
                assert ("training_source" not in kwargs) and ("training_sources" not in kwargs), \
                    "When supplying multiple training_files, the training_file names are used as the training_sources."
                assert "training_file" not in kwargs, \
                    "training_file and training_files cannot be supplied simultaneously."
                self.training_files = kwargs["training_files"]
                self.training_file = None
                self.training_source = None
            else:
                assert "training_files" not in kwargs, \
                    "training_file and training_files cannot be supplied simultaneously."
                self.training_files = None
                self.training_file = kwargs["training_file"]
                self.training_source = kwargs["training_source"]

            # Only (validation_file and validation_source) or (validation_files) can be specified, but
            # not both simultaneously.
            if "validation_files" in kwargs:
                # Same note as above about "validation_sources"
                assert ("validation_source" not in kwargs) and ("validation_sources" not in kwargs), \
                    "When supplying multiple validation_files, the validation file names are used as the validation_sources."
                assert "validation_file" not in kwargs, \
                    "validation_file and validation_files cannot be supplied simultaneously."
                self.validation_files = kwargs["validation_files"]
                self.validation_file = None
                self.validation_source = None
            else:
                assert "validation_files" not in kwargs, \
                    "validation_file and validation_files cannot be supplied simultaneously."
                
                if "validation_file" not in kwargs:
                    # assume no validation_files are specified and we either
                    # want to use cross-validation on the training_files, or
                    # we're just going to ignore validation altogether.
                    self.validation_files = None
                    self.validation_file = None
                    self.validation_source = None
                else:
                    self.validation_files = None
                    self.validation_file = kwargs["validation_file"]
                    self.validation_source = kwargs["validation_source"]

            self.epochs     = epochs
            self.batch_size = batch_size
            self.epoch_size = epoch_size
            self.optimizer  = optimizer
            if learning_rate_scheduler == 'default':
                learning_rate_scheduler = lambda epoch: 0.001 if epoch < epochs//2 else 0.0001
            self.learning_rate_scheduler = learning_rate_scheduler

            self.number_of_validation_sequences = number_of_validation_sequences
            self.max_training_samples = max_training_samples

    """
    A class which bundles the save parameters of a PredNetClient object.
    """
    class SaveParameters(object):
        
        def __init__(self,
                     save_weights=True,
                     weights_file=None,
                     save_model=True,
                     model_file=None,
                     save_best_only=True,
                     checkpoint_monitor='val_loss',
                     checkpoint_period=1):
            self.save_weights = save_weights
            self.save_model   = save_model

            self.weights_file = weights_file
            self.model_file   = model_file
        
            self.save_best_only     = save_best_only
            self.checkpoint_monitor = checkpoint_monitor
            self.checkpoint_period  = checkpoint_period
        
        def _fix_file_names(self, model_name):
            if self.weights_file is None:
                self.weights_file = os.path.join(model_name, 'weights.hdf5')
            if self.model_file is None:
                self.model_file = os.path.join(model_name, 'model.json')

    @property
    def input_spatial_dimensions(self):       return self.model_params.input_spatial_dimensions
    @property
    def input_channels(self):                 return self.model_params.input_channels
    @property
    def number_of_layers(self):               return self.model_params.number_of_layers
    @property
    def A_filter_sizes(self):                 return self.model_params.A_filter_sizes
    @property
    def A_hat_filter_sizes(self):             return self.model_params.A_hat_filter_sizes
    @property
    def R_filter_sizes(self):                 return self.model_params.R_filter_sizes
    @property
    def number_of_filters_per_layer(self):    return self.model_params.number_of_filters_per_layer
    @property
    def loss_weights_per_layer(self):         return self.model_params.loss_weights_per_layer
    @property
    def number_of_timesteps(self):            return self.model_params.number_of_timesteps
    @property
    def loss_weights_per_timestep(self):      return self.model_params.loss_weights_per_timestep
    @property
    def channels_first(self):                 return self.model_params.channels_first

    @property
    def training_file(self):                  return self.train_params.training_file
    @property
    def training_source(self):                return self.train_params.training_source
    @property
    def training_files(self):                 return self.train_params.training_files
    @property
    def validation_file(self):                return self.train_params.validation_file
    @property
    def validation_source(self):              return self.train_params.validation_source
    @property
    def validation_files(self):               return self.train_params.validation_files
    @property
    def epochs(self):                         return self.train_params.epochs
    @property
    def batch_size(self):                     return self.train_params.batch_size
    @property
    def epoch_size(self):                     return self.train_params.epoch_size
    @property
    def optimizer(self):                      return self.train_params.optimizer
    @property
    def learning_rate_scheduler(self):        return self.train_params.learning_rate_scheduler
    @property
    def number_of_validation_sequences(self): return self.train_params.number_of_validation_sequences
    @property
    def max_training_samples(self):           return self.train_params.max_training_samples
    
    @property
    def save_weights(self):                   return self.save_params.save_weights
    @property
    def save_model(self):                    return self.save_params.save_model
    @property
    def weights_file(self):                   return self.save_params.weights_file
    @property
    def model_file(self):                     return self.save_params.model_file
    @property
    def save_best_only(self):                 return self.save_params.save_best_only
    @property
    def checkpoint_monitor(self):             return self.save_params.checkpoint_monitor
    @property
    def checkpoint_period(self):              return self.save_params.checkpoint_period

    @property
    def input_shape(self):
        if self.channels_first:
            return (self.input_channels, ) + self.input_spatial_dimensions
        return self.input_spatial_dimensions + (self.input_channels, )

    def __init__(self, name, train_params, model_params=None, save_params=None):
        self.name = name
        self.train_params = train_params
        # Note: PredNetClient.TrainingParameters is the only class which _requires_ arguments, even if they're
        # still key-word arguments (these are the training files, which must be specified).
        self.model_params = model_params if model_params is not None else PredNetClient.ModelParameters()
        self.save_params = save_params if save_params is not None else PredNetClient.SaveParameters()

        # Weight and model file names cannot be None, but the default names depend on the model name.
        self.save_params._fix_file_names(self.name)

        self.model = None
        self.built = False

        self._loaded_from_file = False

    """
    Load a PredNet model from a file.
    """
    def load_from_file(self, model_file, json_file):
        self._loaded_from_file = True

        raise NotImplementedError("This feature hasn't been implemented yet.")

    """
    Build the actual PredNet model object.
    """
    def build_model(self):
        loss_weights_per_layer    = np.expand_dims(self.loss_weights_per_layer, 1)
        loss_weights_per_timestep = np.expand_dims(self.loss_weights_per_timestep, 1)

        self.prednet_model = PredNet(
            self.number_of_filters_per_layer,
            self.number_of_filters_per_layer, # NOTE: As far as I can tell, the R_stack_sizes
                                              # must equal the stack_sizes, no?
            self.A_filter_sizes,
            self.A_hat_filter_sizes,
            self.R_filter_sizes,
            output_mode='error',
            return_sequences=True)

        inputs = Input(shape=(self.number_of_timesteps, ) + self.input_shape)

        # The output will have shape (batch_size, nt, nb_layers). The outputs correspond
        # to the errors at each time step and layer.
        errors = self.prednet_model(inputs)

        errors_by_time = TimeDistributed(
            Dense(1, trainable=False),
            weights=[loss_weights_per_layer, np.zeros(1)],
            trainable=False
        )(errors)

        errors_by_time = Flatten()(errors_by_time)

        final_errors = Dense(
            1,
            weights=[loss_weights_per_timestep, np.zeros(1)],
            trainable=False
        )(errors_by_time)
        
        self.model = Model(inputs=inputs, outputs=final_errors)
        self.model.compile(loss='mean_absolute_error', optimizer=self.optimizer)
        
        self.model_built = True

    def fit(self, test_mode=True):
        if not self.model_built:
            raise Exception("PredNetClient model must be built first.")

        if self.training_files is None:
            training_generator = SequenceGenerator(
                self.number_of_timesteps,
                batch_size=self.batch_size,
                shuffle=True,
                max_training_samples=self.max_training_samples,
                data_file=self.training_file,
                source_file=self.training_source
            )
        else:
            training_generator = SequenceGenerator(
                self.number_of_timesteps, 
                batch_size=self.batch_size, 
                shuffle=True,
                max_training_samples=self.max_training_samples,
                data_files=self.training_files
            )

        if self.validation_files is None:
            val_generator = SequenceGenerator(
                self.number_of_timesteps,
                batch_size=self.batch_size,
                N_seq=self.number_of_validation_sequences,
                max_training_samples=self.max_training_samples,
                data_file=self.validation_file,
                source_file=self.validation_source
            )
        elif self.validation_files is not None:
            val_generator = SequenceGenerator(
                self.number_of_timesteps,
                batch_size=self.batch_size,
                N_seq=self.number_of_validation_sequences,
                max_training_samples=self.max_training_samples,
                data_files=self.validation_files
            )
        else:
            # Ignore validation altogether
            val_generator = None

        lr_scheduler = None
        if isinstance(self.learning_rate_scheduler, LearningRateScheduler):
            lr_scheduler = self.learning_rate_scheduler
        else:
            lr_scheduler = LearningRateScheduler(self.learning_rate_scheduler)
        self.callbacks = [lr_scheduler]

        if self.save_weights:
            dir_name = os.path.dirname(self.weights_file)
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            self.callbacks.append(
                ModelCheckpoint(
                    filepath=self.weights_file,
                    monitor=self.checkpoint_monitor,
                    save_best_only=self.save_best_only,
                    period=self.checkpoint_period
                )
            )

        if test_mode:
            return

        if val_generator is None:
            self.history = self.model.fit_generator(
                training_generator,
                self.epoch_size / self.batch_size,
                self.epochs,
                callbacks=self.callbacks
            )
        else:
            self.history = self.model.fit_generator(
                training_generator,
                self.epoch_size / self.batch_size,
                self.epochs,
                callbacks=self.callbacks,
                validation_data=val_generator,
                validation_steps=self.number_of_validation_sequences / self.batch_size
            )

        if self.save_model:
            model_as_json = self.model.to_json()
            with open(self.model_file, "w") as file:
                file.write(model_as_json)

    """
    Split the given data into a training set and a validation set.
    """
    def cross_validate(self, validation_split=0.1):
        assert self.training_files is not None, \
            "In order to cross-validate training data, training_files must be supplied."
        assert self.validation_files is None, \
            "validation_files have already been chosen."
        N = len(self.training_files)
        validation_files = []
        for i in range(int(validation_split * N)):
            # At the ith validation file we have removed i files from the training_files
            # list, so it' length will be it's original length minus i.
            j = random.randint(0, N - 1 - i)
            validation_files.append(self.training_files[j])
            self.training_files.remove(j)
        self.train_params.validation_files = validation_files

    def visualize_dimensionality(self):
        pass

        #
        # Methods
        # =======
        # Must-haves
        # ----------
        # fit
        # evaluate
        # load_from_file
        #
        # Nice-to-haves
        # -------------
        # cross_validate
        #   - automatically split the given training_files into a "training set" and a "validation set"
        # visualize_dimensionality
        #   - prints a nicely formatted diagram showing the dimensions of data as it passes through
        #     the forward and backward pass

if __name__ == "__main__":
    training_params = PredNetClient.TrainingParameters(
        training_file        ='./kitti_data/X_train.hkl',
        training_source      ='./kitti_data/sources_train.hkl',
        validation_file      = './kitti_data/X_val.hkl',
        validation_source    = './kitti_data/sources_val.hkl',
        max_training_samples = 30000
    )

    kitti_prednet = PredNetClient("kitti_prednet", training_params)
    kitti_prednet.build_model()

    start_time = get_Time()

    # kitti_prednet.fit(test_mode=False)

    try:
         kitti_prednet.fit(test_mode=False)
    except Exception as e:
         errorTextSend(e.message)

    doneTextSend(start_time, get_Time(), "Training kitti_prednet")