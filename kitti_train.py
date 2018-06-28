'''
Train PredNet on KITTI sequences. (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
'''

import os
import numpy as np
np.random.seed(123)
from six.moves import cPickle

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam

from prednet import PredNet
from data_utils import SequenceGenerator
from kitti_settings import *

def main(verbose=False):
    # Michael Ala 28/6/2018
    # When using the Tensorflow backend, running the model throws an error saying "You must 
    # feed a value for placeholder tensor 'dense_2_sample_weights' with dtype float and shape 
    # [?]" (this is the last Dense layer in the model). 
    # 
    # According to https://github.com/keras-team/keras/issues/2310, a potential workaround is as follows:
    if K._BACKEND == 'tensorflow':
        from tensorflow import constant as tf_constant
        K._LEARNING_PHASE = tf_constant(0)


    save_model = True  # if weights will be saved
    weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.hdf5')  # where weights will be saved
    json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')

    # Data files
    train_file = os.path.join(DATA_DIR, 'X_train.hkl')
    train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')
    val_file = os.path.join(DATA_DIR, 'X_val.hkl')
    val_sources = os.path.join(DATA_DIR, 'sources_val.hkl')

    # Training parameters
    nb_epoch = 150
    batch_size = 4
    samples_per_epoch = 500
    N_seq_val = 100  # number of sequences to use for validation

    # Model parameters
    n_channels, im_height, im_width = (3, 128, 160)
    input_shape = (n_channels, im_height, im_width) if K.image_data_format() == 'channels_first' else (im_height, im_width, n_channels)
    stack_sizes = (n_channels, 48, 96, 192)
    R_stack_sizes = stack_sizes
    A_filt_sizes = (3, 3, 3)
    Ahat_filt_sizes = (3, 3, 3, 3)
    R_filt_sizes = (3, 3, 3, 3)
    layer_loss_weights = np.array([1., 0., 0., 0.])  # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
    layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
    nt = 10  # number of timesteps used for sequences in training
    time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))  # equally weight all timesteps except the first
    time_loss_weights[0] = 0


    prednet = PredNet(stack_sizes, R_stack_sizes,
                    A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                    output_mode='error', return_sequences=True)

    inputs = Input(shape=(nt,) + input_shape)

    # The output will have shape (batch_size, nt, nb_layers). The outputs correspond
    # to the errors at each time step and layer.
    errors = prednet(inputs)

    # This merely computes a weighted sum of the errors layer by layer throughout time.
    # The output has shape (batch_size, nt, 1).
    errors_by_time = TimeDistributed(
        Dense(1, trainable=False), 
        weights=[layer_loss_weights, np.zeros(1)], 
        trainable=False)(errors) 

    # Will have shape (batch_size, nt)
    errors_by_time = Flatten()(errors_by_time)


    # The output of this final layer is the weighted sum over time of the weighted
    # sums of the errors layer-by-layer, which is the final L_train function from
    # the original paper.
    final_errors = Dense(
        1, 
        weights=[time_loss_weights, np.zeros(1)], 
        trainable=False)(errors_by_time)

    model = Model(inputs=inputs, outputs=final_errors)
    model.compile(loss='mean_absolute_error', optimizer='adam', sample_weight_mode='temporal')

    if verbose:
        model.summary()

    train_generator = SequenceGenerator(train_file, train_sources, nt, batch_size=batch_size, shuffle=True)
    val_generator = SequenceGenerator(val_file, val_sources, nt, batch_size=batch_size, N_seq=N_seq_val)

    lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
    callbacks = [LearningRateScheduler(lr_schedule)]
    if save_model:
        if not os.path.exists(WEIGHTS_DIR): os.mkdir(WEIGHTS_DIR)
        callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True))

    history = model.fit_generator(train_generator, samples_per_epoch / batch_size, nb_epoch, callbacks=callbacks,
                    validation_data=val_generator, validation_steps=N_seq_val / batch_size)

    if save_model:
        json_string = model.to_json()
        with open(json_file, "w") as f:
            f.write(json_string)

if __name__ == "__main__":
    import sys
    args = sys.argv

    verbose = '--verbose' in args
    main(verbose=verbose)
