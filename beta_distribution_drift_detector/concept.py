import numpy as np
from beta_distribution_drift_detector.beta_distribution import BetaDistribution


class Concept:
    """ Class that models the current concept of an environment by the beta distributed error rate of a classifier
    """

    def __init__(self):
        self._error_distribution = None

    def init_distribution(self, error_counts, all_counts, classes=None):
        """ Init the distribution

        :param error_counts: Number of observed misclassifications of the classifier
        :param all_counts: Number of observations
        :param classes: optional, classes of the environment
        :return:
        """
        self._error_distribution = BetaDistribution(error_counts, all_counts - error_counts)

    def update(self, error_counts, all_counts, decay_shape_a, decay_shape_b):
        """ Update the concept given new observations

        :param error_counts: Number of new observed misclassifications of the classifier
        :param all_counts: Number of new observations
        :param decay_shape_a: Shape parameter a of the function that defines the decay for old observations
        :param decay_shape_b: Shape parameter b of the function that defines the decay for old observations
        :return:
        """

        # compute the decay for the old parameters
        decay = self._compute_exp_decay(decay_shape_a, decay_shape_b, self._error_distribution.n_updates)

        # update the distribution
        self._error_distribution.update_counts(error_counts, all_counts-error_counts, decay)

    def is_similar(self, error_counts, all_counts, confidence_bound):
        """ Checks whether new observations are likely to belong the previous concept

        :param error_counts: Number of observed misclassifications of the classifier
        :param all_counts: Number of observations
        :param confidence_bound: Confidence bound for the beta distribution
        :return: bool if concepts are similar
        """

        # compute the current error rate
        error_rate = error_counts / all_counts

        # check if the most recent error rate follows the existing beta distribution
        is_confident = self._error_distribution.is_confident(error_rate, confidence_bound)
        return is_confident

    @property
    def n_updates(self):
        return self._error_distribution.n_updates

    @staticmethod
    def _compute_exp_decay(a, b, n_updates):
        """ Function to compute the decay for old observations
        """
        decay = 1 / np.exp((a * (n_updates + b))) + 1.1
        return decay


class ClassBasedConcept(Concept):
    """ Extension of the 'Concept' class:
        Models the current concept based on the error distribution of each single class
    """

    def __init__(self):
        self._classes = None

        super().__init__()

    def init_distribution(self, error_counts, all_counts, classes=None):
        self._classes = classes

        # define one beta distribution for each single class
        self._error_distribution = [BetaDistribution(error_counts[i], all_counts[i]-error_counts[i])
                                    for i in range(len(classes))]

    def update(self, error_counts, all_counts, decay_shape_a, decay_shape_b):
        for c_idx, c in enumerate(self._classes):

            decay = self._compute_exp_decay(decay_shape_a, decay_shape_b, self._error_distribution[c_idx].n_updates)
            self._error_distribution[c_idx].update_counts(error_counts[c_idx],
                                                          all_counts[c_idx]-error_counts[c_idx],
                                                          decay)

    def determine_differing_classes(self, all_classes, error_counts, all_counts, confidence_bound=0.997):
        """ Checks whether one or multiple classes of current observations differ from the previous concept
        """

        tmp_concept_error_rate = error_counts / all_counts

        # check if all current classes are existing in the previous concept
        new_classes = all_classes[~np.isin(all_classes, self._classes)]
        common_classes = all_classes[np.isin(all_classes, self._classes)]

        diff_classes = new_classes
        for tmp_class in common_classes:
            concept_class_idx = np.argwhere(self._classes == tmp_class).item()
            tmp_class_idx = np.argwhere(all_classes == tmp_class).item()

            if not self._error_distribution[concept_class_idx].is_confident(tmp_concept_error_rate[tmp_class_idx],
                                                                            confidence_bound):
                diff_classes = np.hstack((diff_classes, tmp_class))

        return diff_classes

    @property
    def classes(self):
        return self._classes

    @property
    def n_updates(self):
        return min([distribution.n_updates for distribution in self._error_distribution])


