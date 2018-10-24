from collections import Iterable
import hickle as hkl
import numpy as np
from keras import backend as K
from keras.preprocessing.image import Iterator

# Data generator that creates sequences for input into PredNet.
class SequenceGenerator(Iterator):
    def __init__(self, nt, batch_size=8, shuffle=False, seed=None,
                 output_mode='error', sequence_start_mode='all', N_seq=None,
                 data_format=K.image_data_format(), max_training_samples=30000, **kwargs):
        
        # Only (data_file and source_file) or (data_files) can be specified, but
        # not both simultaneously.
        if kwargs.get("data_files") is not None:
            # Note: "source_files" isn't actually an accepted key word parameter, but we guard against it regardless.
            assert kwargs.get("source_file") is None and kwargs.get("source_files") is None, \
                "When supplying multiple data_files, the data_file names are used as the source_files."
            assert kwargs.get("data_file") is None, \
                "data_file and data_files cannot be supplied simultaneously."
        else:
            assert kwargs.get("data_files") is None, \
                "data_file and data_files cannot be supplied simultaneously."

        if "data_files" in kwargs:
            _X = []
            _sources = []
            for file in kwargs["data_files"]:
                file_array = hkl.load(file)
                _X.append(file_array)
                _sources.extend([file] * file_array.shape[0])

            # The elements of _X will be arrays of image arrays, so concatenating
            # along axis=0 (default) simply creates a big array with every image
            # from every data_file listed along the 0th axis.
            self.X = np.concatenate(_X)
            self.sources = _sources
        else:
            self.X = hkl.load(kwargs["data_file"])

            # source for each image so when creating sequences can assure that 
            # consecutive frames are from same video
            self.sources = hkl.load(kwargs["source_file"])

        self.max_training_samples = max_training_samples
        self.X = self.X[:max_training_samples]
        self.sources = self.sources[:max_training_samples]

        self.nt = nt
        self.batch_size = batch_size
        self.data_format = data_format
        assert sequence_start_mode in {'all', 'unique'}, 'sequence_start_mode must be in {all, unique}'
        self.sequence_start_mode = sequence_start_mode
        assert output_mode in {'error', 'prediction'}, 'output_mode must be in {error, prediction}'
        self.output_mode = output_mode

        if self.data_format == 'channels_first':
            self.X = np.transpose(self.X, (0, 3, 1, 2))
        self.im_shape = self.X[0].shape

        if self.sequence_start_mode == 'all':  # allow for any possible sequence, starting from any frame
            self.possible_starts = np.array(
                [i for i in range(self.X.shape[0] - self.nt) 
                    if self.sources[i] == self.sources[i + self.nt - 1]])

        elif self.sequence_start_mode == 'unique':  #create sequences where each unique frame is in at most one sequence
            curr_location = 0
            possible_starts = []
            while curr_location < self.X.shape[0] - self.nt + 1:
                if self.sources[curr_location] == self.sources[curr_location + self.nt - 1]:
                    possible_starts.append(curr_location)
                    curr_location += self.nt
                else:
                    curr_location += 1
            self.possible_starts = possible_starts

        if shuffle:
            self.possible_starts = np.random.permutation(self.possible_starts)
        if N_seq is not None and len(self.possible_starts) > N_seq:  # select a subset of sequences if want to
            self.possible_starts = self.possible_starts[:N_seq]
        self.N_sequences = len(self.possible_starts)
        super(SequenceGenerator, self).__init__(len(self.possible_starts), batch_size, shuffle, seed)

        # Michael Ala 28/6/2018: Not sure if this is how the original creator intended
        # this to be used, but it's how I'm using it now, because the original implementation
        # doesn't work anymore.
        # self.X_all = self.create_all()

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        batch_x = np.zeros((current_batch_size, self.nt) + self.im_shape, np.float32)
        for i, idx in enumerate(index_array):
            idx = self.possible_starts[idx]
            batch_x[i] = self.preprocess(self.X[idx:idx+self.nt])
        if self.output_mode == 'error':  # model outputs errors, so y should be zeros
            batch_y = np.zeros(current_batch_size, np.float32)
        elif self.output_mode == 'prediction':  # output actual pixels
            batch_y = batch_x
        return batch_x, batch_y

    def preprocess(self, X):
        return X.astype(np.float16) / 255

    def create_all(self):
        X_all = np.zeros((self.N_sequences, self.nt) + self.im_shape, np.float16)
        for i, idx in enumerate(self.possible_starts):
            X_all[i] = self.preprocess(self.X[idx:idx+self.nt])
        return X_all

    def _get_batches_of_transformed_samples(self, index_array):
        # The first element of the tuple we're returning is the input batch,
        # a set of sequences of self.nt consecutive images (where the indices 
        # in the given index_array are elements of self.possible_starts, meaning 
        # we never have the issue of fetching a batch of images that belong to
        # two separate videos).
        #
        # The second element of the tuple we're returning represents the labels
        # for each input image. The reason we send in zeros is because the outputs
        # of the model in kitti_train.py is the weighted sum over time, over each
        # layer, of the errors at that time-step and layer, which we wish to minimize,
        # so clearly the target for any input is zero.

        X_batch = np.zeros((len(index_array), self.nt) + self.im_shape, np.float32)
        for i, index in enumerate(index_array):
            X_batch[i] = self.preprocess(self.X[index:index+self.nt])

        return X_batch, np.zeros(len(index_array))
