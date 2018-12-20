

class AEDimReductor:
    """ Dimensionality reduction using a bottleneck Autoencoder

    """

    def __init__(self, base_autoencoder):
        """

        :param base_autoencoder: prototype Autoencoder
        """
        self._autoencoder = base_autoencoder.clone()

    def partial_fit(self, X):
        """ Fit the Autoencoder

        :param X: np.array(n_samples, n_features), Input data
        :return:
        """
        self._autoencoder.partial_fit(X)

    def transform(self, X):
        """ Applies the dimensionality reduction to some input

        :param X: np.array(n_samples, n_features), Input data
        :return: np.array(n_samples, n_low_dims), Embedded data
        """
        code = self._autoencoder.get_code(X)
        return code

    @property
    def n_components(self):
        return self._autoencoder.code_size

    def reset(self):
        self._autoencoder = self._autoencoder.clone(include_weights=False)
