from beta_distribution_drift_detector.bddd import BDDD
from beta_distribution_drift_detector.concept import ClassBasedConcept


class BDDDC(BDDD):
    """ Extension of the Beta Distribution Drift Detector that detects drift based on the class level
        Models on Beta Distribution for the error rate for each class
    """
    def __init__(self, warn_level=0.95, drift_level=0.997,
                 decay_shape_a=0.05, decay_shape_b=-7.0,
                 reset_counts=False,
                 prior_error_rate=None,
                 n_stable_assumed_batches=2):

        super().__init__(warn_level, drift_level,
                         decay_shape_a, decay_shape_b,
                         reset_counts,
                         prior_error_rate,
                         n_stable_assumed_batches)

        self._drifting_classes = []
        self._warning_classes = []

    def add_element(self, prediction, labels=None, classifier_changed=True):
        if labels is None:
            raise ValueError("Class based drift detection expects true class labels, but labels=None")

        # get number of misclassifications for each single class
        all_classes, error_counts, all_counts = self._get_counts(prediction, labels, class_based=True)

        if self._concept is None:
            self._init_concept(all_counts, all_classes)

        # update initial beta distribution if required
        if self._prior_error_rate is None and self._concept.n_updates < self._n_stable_assumed_batches:
            self._concept.update(error_counts, all_counts, self._decay_shape_a, self._decay_shape_b)

        else:
            self._check_for_concept_drift(error_counts, all_counts, all_classes)
            if classifier_changed:
                if not (self._in_drift_zone or self._in_warning_zone):
                    self._concept.update(error_counts, all_counts, self._decay_shape_a, self._decay_shape_b)

    def _init_concept(self, counts, classes=None):
        if self._prior_error_rate is not None:
            prior_error_counts = self._prior_error_rate * counts
        else:
            prior_error_counts = counts / 2

        self._concept = ClassBasedConcept()
        self._concept.init_distribution(prior_error_counts, counts, classes)

    def detected_change(self):
        # signal drift if the error rate of at least one class is above the drift level
        return len(self._drifting_classes) > 0

    def detected_warning_zone(self):
        # signal warning if the error rate of at least one class is above the warn level
        return len(self._warning_classes) > 0

    def _check_for_concept_drift(self, error_counts, all_counts, classes=None):
        # check for classes above warning level
        self._warning_classes = self._concept.determine_differing_classes(classes, error_counts, all_counts,
                                                                          self._warn_level)

        # check for classes above drift level
        self._drifting_classes = self._concept.determine_differing_classes(classes, error_counts, all_counts,
                                                                           self._drift_level)

    @property
    def drifting_classes(self):
        # property to read the drifting classes if a drift got signaled
        return self._drifting_classes

    @property
    def warning_classes(self):
        # property to read the drifting classes if a warning signaled
        return self._warning_classes

