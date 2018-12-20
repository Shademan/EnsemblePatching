from sklearn import clone
from copy import deepcopy


class BaseClassifier:
    """
    Wrapper class that allows creating new instances from scikit-learn and scikit-multiflow classifiers given a
    prototype classifier.
    If other classifiers are used, they need to implement the 'clone()' method and the 'classes' property

    """

    def __init__(self, prototype_clf):
        """
        :param prototype_clf:   scikit-learn, scikit-multiflow, or other estimator that implements the 'clone()' method
                                and the 'classes' property
                                Prototype classifier that gets instantiated in ensemble methods
        """
        self._prototype_package = prototype_clf.__module__.split(".")[0]

        self._clf = None
        self._instantiate_classifier(prototype_clf)

    def fit(self, X, y):
        self._clf.fit(X, y)

    def partial_fit(self, X, y, classes=None, validation_data=None):
        if validation_data is not None:
            self._clf.partial_fit(X, y, classes, validation_data)
        else:
            self._clf.partial_fit(X, y, classes)

    def predict(self, X):
        return self._clf.predict(X)

    def predict_proba(self, X):
        return self._clf.predict_proba(X)

    @property
    def classes(self):
        if self._prototype_package == 'sklearn':
            return self._clf.classes_
        else:
            return self._clf.classes

    def _instantiate_classifier(self, prototype_clf):
        if self._prototype_package == 'sklearn':
            self._clf = clone(prototype_clf)
        elif self._prototype_package == 'skmultiflow':
            self._clf = deepcopy(prototype_clf)
            self._clf.reset()
        else:
            self._clf = prototype_clf.clone()

    def compile(self, **kwargs):
        if hasattr(self._clf, 'compile'):
            self._clf.compile(**kwargs)
        else:
            raise NotImplementedError

    def add_new_classes(self, new_classes):
        if hasattr(self._clf, 'add_new_classes'):
            self._clf.add_new_classes(new_classes)
        else:
            raise NotImplementedError

    @property
    def is_compiled(self):
        if hasattr(self._clf, 'is_compiled'):
            return self._clf.is_compiled
        else:
            raise NotImplementedError

