import numpy as np
from beta_distribution_drift_detector.concept import Concept
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector


class BDDD(BaseDriftDetector):
    """ Beta Distribution Drift Detection proposed in:
        Fleckenstein, Lukas and Kauschke, Sebastian, and Johannes Fürnkranz:
        "Beta Distribution Drift Detection for Adaptive Classifiers." https://arxiv.org/abs/1811.10900.

    """

    def __init__(self, warn_level=0.95, drift_level=0.997,
                 decay_shape_a=0.05, decay_shape_b=-7.0,
                 reset_counts=True,
                 prior_error_rate=None,
                 n_stable_assumed_batches=2):

        """

        :param warn_level: Significance level for waring
        :param drift_level: Significance level for detecting a concept change
        :param decay_shape_a: Shape parameter 'a' in the update rule of the prior decay
        :param decay_shape_b: Shape parameter 'b' in the update rule of the prior decay
        :param reset_counts: Flag that decides whether the error distribution should be reset of detecting a drift
        :param prior_error_rate: optional, the classifiers a-priori assumed error rate
        :param n_stable_assumed_batches: Number of stable assumed batches on which the error distribution is initialized
                                         Required if no prior_error_rate is given
        """

        self._warn_level = warn_level
        self._drift_level = drift_level
        self._decay_shape_a = decay_shape_a
        self._decay_shape_b = decay_shape_b
        self._reset_counts = reset_counts
        self._prior_error_rate = prior_error_rate
        self._n_stable_assumed_batches = n_stable_assumed_batches # necessary if prior_error_rate is not given

        self._concept = None
        self._in_warning_zone = False
        self._in_drift_zone = False


        super().__init__()

    def add_element(self, prediction, labels=None, classifier_changed=True):
        """ Adds elements to the drift detector

        :param prediction: If no labels are given, prediction is expected to be binary with 0=correct classification,
                           1=wrong classification. If labels are given, prediction is the class predicted by the
                           corresponding classifier
        :param labels: True class labels
        :param classifier_changed:  Flag that indicates whether the classifier changed after the last call.
                                    If True, the concept gets updated, otherwise only a check for drift gets executed

        """
        if not np.array_equal(prediction, prediction.astype(bool)) and labels is None:
            raise ValueError("Prediction not binary and labels=None. "
                             "If predictions are not binary, labels are required.")

        # init concept with prior knowledge or random guessing
        if self._concept is None:
            self._init_concept(batch_size=len(prediction))

        error_counts, all_counts = self._get_counts(prediction, labels, class_based=False)

        # update initial beta distribution if required
        if self._prior_error_rate is None and self._concept.n_updates < self._n_stable_assumed_batches:
            self._concept.update(error_counts, all_counts, self._decay_shape_a, self._decay_shape_b)

        else:
            self._check_for_concept_drift(error_counts, all_counts)
            if classifier_changed:
                if not (self._in_drift_zone or self._in_warning_zone):
                    self._concept.update(error_counts, all_counts, self._decay_shape_a, self._decay_shape_b)

    def reset(self):
        self._concept = None

    def detected_change(self):
        return self._in_drift_zone

    def detected_warning_zone(self):
        return self._in_warning_zone

    def _init_concept(self, batch_size, classes=None):
        """
        If prior knowledge about the classifier error exist, init concept with that knowledge.
        Otherwise assume chance prediction
        """

        if self._prior_error_rate is not None:
            prior_error_counts = self._prior_error_rate * batch_size
        else:
            prior_error_counts = batch_size / 2

        self._concept = Concept()
        self._concept.init_distribution(prior_error_counts, batch_size)
        self._in_warning_zone = False
        self._in_drift_zone = False

    def _check_for_concept_drift(self, error_counts, all_counts, classes=None):
        # check for drift
        if not self._concept.is_similar(error_counts, all_counts, self._drift_level):
            self._in_drift_zone = True

            if self._reset_counts:
                self.reset()

        # check for warning
        elif not self._concept.is_similar(error_counts, all_counts, self._warn_level):
            self._in_drift_zone = False
            self._in_warning_zone = True

        else:
            self._in_warning_zone = False
            self._in_drift_zone = False

    def _get_counts(self, prediction, labels, class_based=False):
        # count the number of misclassiciations. If class_based=True count the errors per class
        if np.array_equal(prediction, prediction.astype(bool)) and labels is None:
            error_counts, all_counts = self._get_binary_data_counts(prediction)
        else:
            if class_based:
                classes, error_counts, all_counts = self._get_labeled_data_counts(prediction, labels, class_based)
                return classes, error_counts, all_counts
            else:
                error_counts, all_counts = self._get_labeled_data_counts(prediction, labels, class_based)
        return error_counts, all_counts

    @staticmethod
    def _get_binary_data_counts(prediction):
        # get error counts for binary input streams
        error_counts = np.sum(prediction)
        all_counts = len(prediction)
        return error_counts, all_counts

    def _get_labeled_data_counts(self, prediction, labels, class_based=False):
        # get error counts given the classifiers prediction and true class labels
        error_classes, error_counts = self._count_misclassifications(prediction, labels)
        all_classes, all_counts = np.unique(labels, return_counts=True)

        if not class_based:
            error_counts = np.sum(error_counts)
            return np.sum(error_counts), np.sum(all_counts)

        else:
            # add zero counts for classes without misclassifications
            pad_error_counts = np.zeros(len(all_classes))
            pad_error_counts[np.isin(all_classes, error_classes)] = error_counts
            return all_classes, pad_error_counts, all_counts

    @staticmethod
    def _count_misclassifications(prediction, labels):
        error_idx = prediction != labels
        error_labels = labels[error_idx]
        error_classes, error_counts = np.unique(error_labels, return_counts=True)
        return error_classes, error_counts

    def clone(self):
        new_instance = type(self)(self._warn_level,
                                  self._drift_level,
                                  self._decay_shape_a,
                                  self._decay_shape_b,
                                  self._reset_counts,
                                  self._prior_error_rate,
                                  self._n_stable_assumed_batches)
        return new_instance

    def get_info(self):
        return 'Warn level: ' + str(self._warn_level) + \
               ' - Drift level: ' + str(self._drift_level) + \
               ' - Decay shape a: ' + str(self._decay_shape_a) + \
               ' - Decay shape b: ' + str(self._decay_shape_b)
