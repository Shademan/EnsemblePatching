import numpy as np
from core.classifiers.base_classifier import BaseClassifier


class Conductor:
    """
    Conductor that applies a RNN classifier to the sequence of ensemble member predictions
    """
    def __init__(self, base_rnn_classifier):
        """

        :param base_rnn_classifier: Prototype RNN classifier
        """

        self._rnn_clf = BaseClassifier(base_rnn_classifier)

    def fit(self, sequence, y):
        """ Fit the Conductor

        :param sequence: np.array(n_samples, n_ensemble_members, n_features), Sequence to tbe classified
        :param y: np.array(n_samples), Class labels
        :return:
        """

        # check if new classes appeared in the environment and add new outputs to the RNN classifier if necessary
        if self._rnn_clf.is_compiled:
            self._check_for_new_classes(y)

        self._rnn_clf.partial_fit(sequence, y)

    def predict(self, sequence):
        """ Returns the conductor classification given the ensemble members predictions

        :param sequence: np.array(n_samples, n_ensemble_members, n_features), Sequence to tbe classified
        :return: np.array(n_samples), Prediction
        """

        return self._rnn_clf.predict(sequence)

    def _check_for_new_classes(self, y):
        tmp_classes = np.unique(y)
        new_classes = tmp_classes[~np.isin(tmp_classes, self._rnn_clf.classes)]

        if len(new_classes) != 0:
            self._rnn_clf.add_new_classes(new_classes)
