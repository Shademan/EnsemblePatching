import numpy as np


class LabelEncoder:
    """ Laber encoder if labels are sparse, e.g. [0, 3, 5]
        Needed for neural network classifiers where classification nodes can get added without resorting
        (e.g. initially 3 output nodes trained on [0, 3, 5], but at runtime [1, 2] get added resulting in an output
        layer [0, 3, 5, 1, 2]
    """

    def __init__(self):
        self._classes = None

    # Encode Labels
    def fit(self, y):
        self._classes = np.unique(y)

    def add_classes(self, new_classes):
        self._classes = np.hstack((self._classes, new_classes))

    # Transform original labels into encoding
    def transform(self, y):
        return np.array([np.argwhere(self._classes == sample).item() for sample in y])

    # Transform encoding back to original labels
    def inverse_transform(self, y):
        return self._classes[y]
