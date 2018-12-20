import numpy as np
from skmultiflow.core.base import StreamModel
from patching.batch_window_buffer import BatchWindowBuffer
from sklearn.metrics import accuracy_score
from core.classifiers.base_classifier import BaseClassifier


class Patching(StreamModel):
    """ Patching algorithm proposed in:
        Kauschke, Sebastian, and Johannes Fürnkranz: "Batchwise Patching of Classifiers." AAAI. 2018.

    """

    def __init__(self, base_black_box_clf,
                 base_error_region_clf,
                 base_patch_clf,
                 pretrain_size,
                 window_size=8):

        """

        :param base_black_box_clf:  scikit-learn, scikit-multiflow, or other estimator that is compatible
                                    with the 'BaseClassifier' class
                                    Prototype estimator for the black-box classifier
        :param base_error_region_clf:   scikit-learn, scikit-multiflow, or other estimator that is compatible
                                        with the 'BaseClassifier' class
                                        Prototype estimator for the error-region classifier
        :param base_patch_clf:  scikit-learn, scikit-multiflow, or other estimator that is compatible
                                with the 'BaseClassifier' class
                                Prototype estimator for the patch classifier
        :param pretrain_size:   int
                                Number of samples that are used for the initialization phase
        :param window_size: int
                            Number of batches that are stored in a window
        """

        self._base_black_box_clf = base_black_box_clf
        self._base_error_region_clf = base_error_region_clf
        self._base_patch_clf = base_patch_clf
        self._pretrain_size = pretrain_size
        self._window_size = window_size
        self._batch_window_buffer = BatchWindowBuffer(window_size=self._window_size)

        self._black_box_clf = BaseClassifier(base_black_box_clf)
        self._error_region_clf = None
        self._patch_clf = None
        self._global_sample_count = 0

        super().__init__()

    def partial_fit(self, X, y, classes=None, weight=None):
        """ Partial (incremental) fit the model under the stream learning setting.

        :param X:   numpy array of shape [n_samples,n_features]
                    Training data
        :param y:   numpy array of shape [n_samples]
                    Target values. Will be cast to X's dtype if necessary
        :param classes: numpy array of shape [n_classes], optional
                        Classes of the current environment. Are allowed to change across all calls to partial_fit
        :param weight:  numpy array of shape [n_samples], optional
                        Weights applied to individual samples.
                        If not provided, uniform weights are assumed.
        :return:    self
        """

        window_data, window_labels = self._batch_window_buffer.update_buffer(X, y)

        # fit black-box classifier during the initialization phase
        if self._global_sample_count < self._pretrain_size:
            self._black_box_clf.fit(window_data, window_labels)

        # fit the ensemble consisting of the error-region classifier and the patch classifier afterwards
        else:
            self._fit_ensemble(window_data, window_labels)

        self._global_sample_count += len(X)

    def predict(self, X):
        """ Predicts targets using the model.

        :param X:   numpy array of shape [n_samples,n_features]
                    The matrix of samples one wants to predict the labels for.
        :return:    An array-like with all the predictions for the samples in X.
        """

        if self._global_sample_count <= self._pretrain_size:
            prediction = self._black_box_clf.predict(X)
        else:
            prediction = self._predict_ensemble(X)
        return prediction

    def reset(self):
        """ Resets the model to its initial state.

        :return:    self
        """

        self._error_region_clf = None
        self._patch_clf = None
        self._black_box_clf = BaseClassifier(self._base_black_box_clf)
        self._batch_window_buffer.reset()
        self._global_sample_count = 0

    def score(self, X, y):
        """ Calculate the accuracy for the model in its current state.

        :param X:   numpy array of shape [n_samples,n_features]
                    The matrix of samples one wants to predict the labels for.
        :param y:   numpy array of shape [n_samples]
                    Target values. Will be cast to X's dtype if necessary
        :return:    self
        """

        prediction = self.predict(X)
        return accuracy_score(y, prediction)

    def get_info(self):
        """ Returns the models parameters as string

        :return:    string
                    The model's parameters
        """

        description = type(self).__name__ + ': '
        description += 'base_black_box_classifier - %s, ' % type(self._base_black_box_clf).__name__
        description += 'black_error_region_classifier - %s, ' % type(self._base_error_region_clf).__name__
        description += 'base_patch_classifier - %s, ' % type(self._base_patch_clf).__name__
        description += 'base_patch_classifier - %s, ' % type(self._base_patch_clf).__name__
        description += 'pretrain_size - %s, ' % str(self._pretrain_size)
        description += 'window_size - %s' % str(self._window_size)
        return description

    def _fit_ensemble(self, X, y):
        self._fit_error_region_classifier(X, y)
        self._fit_patch(X, y)

    def _fit_error_region_classifier(self, X, y):
        # determine the true error regions given some labeled data
        error_region_labels = self._determine_error_region_labels(X, y)

        # train new error-region classifier at each time step
        self._error_region_clf = BaseClassifier(self._base_error_region_clf)
        self._error_region_clf.fit(X, error_region_labels)

    def _fit_patch(self, X, y):
        # determine the data for which a patch is necessary
        patch_data, patch_labels = self._determine_patch_data(X, y)
        if len(patch_labels) == 0:
            return

        # train new patch classifier at each time step
        self._patch_clf = BaseClassifier(self._base_patch_clf)
        self._patch_clf.fit(patch_data, patch_labels)

    def _determine_patch_data(self, X, y):
        # build the patch data by selecting samples for which the black-box classifiers prediction is wrong
        prediction_base = self._black_box_clf.predict(X)
        error_region_samples = (prediction_base != y)
        patch_data = X[error_region_samples]
        patch_labels = y[error_region_samples]
        return patch_data, patch_labels

    def _determine_error_region_labels(self, X, y):
        # determine binary labels indicating if the black-box classifier errs or not
        prediction_base = self._black_box_clf.predict(X)
        error_region_labels = (prediction_base != y)
        return error_region_labels

    def _predict_ensemble(self, X):
        # predict the error regions using the error-region classifier
        pred_error_regions = self._error_region_clf.predict(X)
        sample_idx_stable = np.nonzero(~pred_error_regions)[0]
        sample_idx_error_region = np.nonzero(pred_error_regions)[0]

        prediction = np.zeros(len(X))
        # call the black-box classifier on samples which are classified to be outside of the error region
        if len(X[sample_idx_stable]) != 0:
            prediction[sample_idx_stable] = self._black_box_clf.predict(X[sample_idx_stable])

        # call the patch classifier on samples which are classified to be inside of the error region
        if len(X[sample_idx_error_region]) != 0:
            prediction[sample_idx_error_region] = self._patch_clf.predict(X[sample_idx_error_region])
        return prediction

    def fit(self, X, y, classes=None, weight=None):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError


