from keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping
from core.label_encoder import LabelEncoder
from core.keras_utils import clone_optimizer


class GRUClassifier:
    """ GRU Classifier using Keras. Consists of a single GRU layer and a softmax classification layer

    """

    def __init__(self, n_units, base_optimizer='adam', epochs=100, batch_size=64, validation_split=0.2):
        """

        :param n_units: Number of units in the GRU layer
        :param base_optimizer: Keras optimizer
        :param epochs:  Number of epochs during optimization.
                        If 'validation_split' > 0 early stopping is applied and the given epochs are the maximal number
        :param batch_size: Batch size applied during optimization
        :param validation_split: Fraction for splitting the data in train and validation for applying early stopping
        """

        self._n_units = n_units
        self._base_optimizer = base_optimizer
        self._epochs = epochs
        self._batch_size = batch_size
        self._validation_split = validation_split

        self._model = Sequential()
        if validation_split > 0:
            self._callbacks = [EarlyStopping(verbose=0, patience=5)]
        else:
            self._callbacks = None

        self._is_compiled = False
        self._label_encoder = LabelEncoder()

        self._classes = None
        self._n_features = None
        self._optimizer = None

    def partial_fit(self, X, y, classes=None):
        """ Fit the GRU classifier

        :param X: np.array (n_samples, sequence_length, n_features), Input data
        :param y: np.array (n_samples), Input labels
        :param classes: ignored
        :return:
        """

        if not self._is_compiled:
            self._compile(n_features=np.shape(X)[2],
                          classes=np.unique(y))

        labels_encode = self._label_encoder.transform(y)

        self._model.fit(X, labels_encode, verbose=0,
                        epochs=self._epochs, batch_size=self._batch_size,
                        validation_split=self._validation_split, callbacks=self._callbacks)

    def predict(self, X):
        """ Classifies some given data

        :param X: np.array (n_samples, sequence_length, n_features), Input data
        :return: np.array(n_samples), Classification result
        """

        prediction_one_hot = self._model.predict(X, verbose=0)
        prediction_dense = self._conv_prediction_to_dense(prediction_one_hot)
        return prediction_dense

    def _compile(self, n_features, classes):
        self._classes = classes
        self._n_features = n_features
        self._optimizer = clone_optimizer(self._base_optimizer)

        self._build_architecture()
        self._model.compile(optimizer=self._optimizer, loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])
        self._is_compiled = True
        self._label_encoder.fit(classes)

    def _build_architecture(self):
        self._model.add(GRU(self._n_units, input_shape=(None, self._n_features), return_sequences=False))
        self._model.add(Dense(len(self._classes), activation='softmax', name='clf_layer'))

    def _conv_prediction_to_dense(self, prediction):
        prediction_dense = np.argmax(prediction, axis=1).flatten()
        prediction_decode = self._label_encoder.inverse_transform(prediction_dense)
        return prediction_decode

    def add_new_classes(self, new_classes):
        """ Adds new classes to the output layer keeping weights of the other output nodes

        :param new_classes: New classes to be added
        :return:
        """

        # Increment the number of outputs
        self._classes = np.hstack((self._classes, new_classes))
        self._label_encoder.add_classes(new_classes)

        weights = self._model.get_layer('clf_layer').get_weights()
        # Adding new weights, weights will be 0 and the connections random
        num_inputs = weights[0].shape[0]
        # init new node weights
        n_new_outputs = len(new_classes)
        new_node_weights = -0.0001 * np.random.random_sample((num_inputs, n_new_outputs)) + 0.0001
        # init new bias weights
        new_bias_weights = np.zeros(n_new_outputs)

        # concat new and old weights
        weights[0] = np.concatenate((weights[0], new_node_weights), axis=1)
        weights[1] = np.concatenate((weights[1], new_bias_weights), axis=0)
        # Deleting the old output layer
        self._model.pop()
        self._model.outputs = [self._model.layers[-1].output]
        self._model.layers[-1].outbound_nodes = []

        # New output layer
        self._model.add(Dense(len(self._classes), activation='softmax', name='clf_layer'))
        # set weights to the layer
        self._model.get_layer('clf_layer').set_weights(weights)

        self._model.compile(optimizer=self._optimizer, loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])

    def clone(self):
        return type(self)(self._n_units,
                          self._base_optimizer,
                          self._epochs,
                          self._batch_size,
                          self._validation_split)

    @property
    def classes(self):
        return self._classes

    @property
    def is_compiled(self):
        return self._is_compiled

