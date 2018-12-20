from scipy.stats import beta


class BetaDistribution:
    """
    Implements a beta distribution Beta(alpha, beta)
    """
    def __init__(self, init_alpha, init_beta):
        self._alpha = init_alpha
        self._beta = init_beta
        self._n_updates = 0

    def update_counts(self, new_alpha, new_beta, decay):
        """
        Updates the shape parameters given some new observations and a decay for the existing ones

        :param new_alpha: Number of new observed ones ('heads')
        :param new_beta: Number of new observed zeros ('tails)
        :param decay: Integer to decay the old observations
        :return:
        """

        self._alpha = self._alpha / decay + new_alpha
        self._beta = self._beta / decay + new_beta
        self._n_updates += 1

    def is_confident(self, pi, percentage):
        """ Checks whether a given samples is confident with the distribution given a significance level

        :param pi: observed sample
        :param percentage: significance level
        :return: bool
        """

        # Compute the upper bound that contains 'percentage' percent of the distribution
        upper_confidence_bound = beta.interval(percentage, self._alpha, self._beta)[1]
        return pi < upper_confidence_bound

    @property
    def n_updates(self):
        return self._n_updates
