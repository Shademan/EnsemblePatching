from keras.layers import *
from sklearn import preprocessing
from keras.regularizers import l2
from keras.initializers import *
from keras.optimizers import *
from keras.models import Model
from keras.models import load_model
import keras.backend as K
from core.keras_utils import clone_optimizer


class AutoEncoder:
    """ Symmetric fully connected Autoencoder using Keras
        (Decoder is a mirrored version of the encoder)

    """
    def __init__(self, encoding_layers,
                 encoding_activations,
                 dropout=0.3,
                 regularizer=None,
                 kernel_initializer='Zeros',
                 bias_initializer=lecun_uniform(),
                 base_optimizer='adam',
                 batch_size=100,
                 epochs=100,
                 normalize=True):
        """

        :param encoding_layers: list, number of nodes per layer in the encoding
        :param encoding_activations: Keras activation, activation function applied in the dense layers
        :param dropout: int, drop rate for drop out layers between the dense layers
        :param regularizer: optional, Keras regularizer for the dense layers
        :param kernel_initializer: Keras initializer for the dense layer kernels
        :param bias_initializer: Keras initializer for the dense layer biases
        :param base_optimizer: Keras optimizer
        :param batch_size: Batch size applied during optimization
        :param epochs: Number of epochs for each optimization step. (Max. number if early stopping is applied)
        :param normalize: Flag if the data should be normalized in range [0, 1]
        """

        self._encoding_layers = encoding_layers
        self._encoding_activations = encoding_activations
        if len(encoding_layers) != len(encoding_activations):
            raise ValueError("Number of encoding activations must match the number of encoding layers. "
                             "Got %i layers but %i activations" % (len(encoding_layers), len(encoding_activations)))

        self._dropout = dropout
        self._regularizer = regularizer
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._base_optimizer = base_optimizer
        self._batch_size = batch_size
        self._epochs = epochs
        self._normalize = normalize

        self._is_compiled = False
        self._optimizer = None
        self._model = None
        self._num_features = None
        self._metrics = None

    def partial_fit(self, data):
        """ Fit the Autoencoder unsupervised on some input data

        :param data: np.array(n_samples, n_features), Input data
        :return:
        """
        if not self._is_compiled:
            self._num_features = np.shape(data)[1]
            self._compile()

        if self._normalize:
            data = data / np.max(np.abs(data))  # normalize data
        self._model.fit(data, data, verbose=0, batch_size=self._batch_size, epochs=self._epochs)

    def predict(self, data):
        """ Predict the reconstruction given some input

        :param data: np.array(n_samples, n_features), Input data
        :return: np.array(n_samples, n_features), Reconstruction of the input data
        """

        if self._normalize:
            norm_factor = np.max(np.abs(data))
        else:
            norm_factor = 1
        pred_reconstruction = self._model.predict(data / norm_factor, verbose=0)
        return pred_reconstruction * norm_factor

    def get_code(self, X):
        """ Returns the code (activation of the bottleneck layer) for some input

        :param X: np.array(n_samples, n_features), Input data
        :return: np.array(n_samples, code_size), Embedded features
        """
        if self._normalize:
            X = X / np.max(np.abs(X))

        # see https://github.com/philipperemy/keras-activations/blob/master/keract/keract.py
        inp = [self._model.input]

        output = self._model.get_layer('encoding').output
        func = K.function(inp + [K.learning_phase()], [output])

        list_inputs = [X, 0.]
        activation = func(list_inputs)[0]
        return activation

    def _compile(self):
        self._build_architecture()
        self._optimizer = clone_optimizer(self._base_optimizer)
        self._model.compile(optimizer=self._optimizer,
                            loss='mean_squared_error')

        self._is_compiled = True

    def _build_architecture(self):
        inputs = Input(shape=(self._num_features, ))
        encoding = self._build_encoding(inputs)
        decoding = self._build_decoding(encoding)
        reconstruction = self._build_reconstruction(decoding)

        self._model = Model(inputs=inputs, outputs=[reconstruction])

    def _build_encoding(self, x):
        for l in range(len(self._encoding_layers)):
            if l == len(self._encoding_layers) - 1:
                layer_name = "encoding"
            else:
                layer_name = None
            x = Dense(self._encoding_layers[l], activation=self._encoding_activations[l],
                      kernel_regularizer=self._regularizer,
                      kernel_initializer=self._kernel_initializer,
                      bias_initializer=self._bias_initializer,
                      name=layer_name)(x)
            x = Dropout(self._dropout)(x)
        return x

    def _build_decoding(self, x):
        # build decoding layers by mirroring the encoder
        decoding_layers = self._encoding_layers[:-1]
        decoding_layers.reverse()

        decoding_activations = self._encoding_activations[:-1]
        decoding_activations.reverse()

        for l in range(len(decoding_layers)):
            x = Dense(decoding_layers[l], activation=decoding_activations[l],
                      kernel_regularizer=self._regularizer,
                      kernel_initializer=self._kernel_initializer,
                      bias_initializer=self._bias_initializer)(x)
            x = Dropout(self._dropout, seed=42)(x)
        return x

    def _build_reconstruction(self, x):
        x = Dense(self._num_features, activation='linear',
                  kernel_regularizer=self._regularizer,
                  kernel_initializer=self._kernel_initializer,
                  bias_initializer=self._bias_initializer,
                  name='reconstruction')(x)
        return x

    @property
    def code_size(self):
        return self._encoding_layers[-1]

    def clone(self, include_weights=False):
        new_clf = type(self)(self._encoding_layers,
                             self._encoding_activations,
                             self._dropout,
                             self._regularizer,
                             self._kernel_initializer,
                             self._bias_initializer,
                             self._base_optimizer,
                             self._batch_size,
                             self._epochs,
                             self._normalize)

        if include_weights:
            self._model.save("tmp_weights.h5")
            new_clf.model = load_model("tmp_weights.h5")
        return new_clf
