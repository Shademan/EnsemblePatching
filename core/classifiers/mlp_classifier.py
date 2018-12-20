from keras.models import Sequential
from keras.layers import *
from keras.models import load_model
from keras.initializers import *
from keras.callbacks import *
from sklearn import preprocessing
from core.keras_utils import clone_optimizer


class MLPClassifier:
    """ Multilayer Perceptron Classifier using Keras

    """
    def __init__(self, layers,
                 activation='relu',
                 dropout=0.5,
                 regularizer=None,
                 kernel_initializer='Zeros',
                 bias_initializer=lecun_uniform(),
                 base_optimizer='adam',
                 batch_size=64,
                 epochs=100):
        """

        :param layers: list, number of nodes per layer
        :param activation: Keras activation, activation function applied in the dense layers
        :param dropout: int, drop rate for drop out layers between the dense layers
        :param regularizer: optional, Keras regularizer for the dense layers
        :param kernel_initializer: Keras initializer for the dense layer kernels
        :param bias_initializer: Keras initializer for the dense layer biases
        :param base_optimizer: Keras optimizer
        :param batch_size: Batch size applied during optimization
        :param epochs: Number of epochs for each optimization step. (Max. number if early stopping is applied)
        """

        self._layers = layers
        self._activation = activation
        self._dropout = dropout
        self._regularizer = regularizer
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._base_optimizer = base_optimizer
        self._batch_size = batch_size
        self._epochs = epochs

        self._model = Sequential()
        self._label_encoder = preprocessing.LabelEncoder()
        self._is_compiled = False
        self._optimizer = None
        self._callbacks = None

    def partial_fit(self, X, y, classes=None, validation_data=None):
        """ Fit the MLP classifier

             :param X: np.array (n_samples, n_features), Input data
             :param y: np.array (n_samples), Input labels
             :param classes: ignored
             :param validation_data: validation data for applying early stopping
             :return:
        """

        if not self._is_compiled:
            if validation_data is not None:
                self._callbacks = [EarlyStopping(verbose=0, patience=5)]

            self._label_encoder.fit(classes)
            self._compile(len(classes))
            self._is_compiled = True

        labels_train = self._label_encoder.transform(y)
        if validation_data is not None:
            labels_val = self._label_encoder.transform(validation_data[1])
            validation_data = (validation_data[0], labels_val)

        self._model.fit(X, labels_train,
                        epochs=self._epochs, callbacks=self._callbacks,
                        validation_data=validation_data,
                        verbose=0)

    def predict(self, X):
        """ Classifies some given data

        :param X: np.array(n_samples, n_features), Input data
        :return: np.array(n_samples), Classification result
        """

        prediction = self._model.predict(X)
        prediction_dense = np.argmax(prediction, axis=1)
        prediction_decode = self._label_encoder.inverse_transform(prediction_dense)
        return prediction_decode

    def _compile(self, n_classes):
        self._build_architecture(n_classes)
        self._optimizer = clone_optimizer(self._base_optimizer)
        self._model.compile(optimizer=self._optimizer,
                            loss='sparse_categorical_crossentropy')

    def _build_architecture(self, n_outputs):
        # stack layers
        for nodes in self._layers:
            self._model.add(Dense(nodes,
                                  activation=self._activation,
                                  kernel_regularizer=self._regularizer,
                                  kernel_initializer=self._kernel_initializer,
                                  bias_initializer=self._bias_initializer))

            self._model.add(Dropout(self._dropout))

        # add classification layer
        self._model.add(Dense(n_outputs,
                              activation='softmax',
                              kernel_regularizer=self._regularizer,
                              kernel_initializer=self._kernel_initializer,
                              bias_initializer=self._bias_initializer))

    def clone(self, include_weights=False):
        new_clf = type(self)(self._layers,
                             self._activation,
                             self._dropout,
                             self._regularizer,
                             self._kernel_initializer,
                             self._bias_initializer,
                             self._base_optimizer,
                             self._batch_size,
                             self._epochs)

        if include_weights:
            self._model.save("tmp_weights.h5")
            new_clf.model = load_model("tmp_weights.h5")
        return new_clf
