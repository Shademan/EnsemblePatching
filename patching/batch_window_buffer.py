import numpy as np


class BatchWindowBuffer:
    def __init__(self, window_size, exclude_first_batch=True):
        """ Buffer that collects a fixed number of batches following the FIFO scheme

        :param window_size: int
                            The number of batches to be collected in the buffer

        :param exclude_first_batch: bool
                                    Flag if the first batch should be excluded from the buffer.
                                    Can be used if the first batch of data corresponds to a pretrain phase
        """

        self._window_size = window_size
        self._exclude_first_batch = exclude_first_batch
        self._buffer_data = []
        self._buffer_labels = []
        self._first_batch = True

    def update_buffer(self, X, y):
        """ Adds a new batch to the buffer and returns the buffers content

        :param X:   numpy array of shape [n_samples,n_features]
                    New training data
        :param y:   numpy array of shape [n_samples]
                    New target values.
        :return:    numpy arrays of shape [current_window_size*n_samples,n_features], [current_window_size*n_samples]
                    The buffered data and the buffered labels
        """

        if self._exclude_first_batch and self._first_batch:
            self._first_batch = False
            return X, y

        else:
            if len(self._buffer_data) < self._window_size:
                self._append_batch(X, y)

            else:
                self._append_batch(X, y)
                self._remove_oldest_batch()
            return self._stack_batches()

    def reset(self):
        self._buffer_data = []
        self._buffer_labels = []
        self._first_batch = True

    def _append_batch(self, X, y):
        self._buffer_data.append(X)
        self._buffer_labels.append(y)

    def _remove_oldest_batch(self):
        del self._buffer_data[0]
        del self._buffer_labels[0]

    def _stack_batches(self):
        data_array = np.vstack(self._buffer_data)
        labels_array = np.vstack(self._buffer_labels)

        if data_array.ndim == 2:
            labels_array = labels_array.flatten()
        return data_array, labels_array
