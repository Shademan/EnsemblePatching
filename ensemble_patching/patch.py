from core.classifiers.base_classifier import BaseClassifier


class Patch:
    """
    Patch that covers some classes for which the black-box classifier errs
    """

    def __init__(self, base_clf, classes, base_drift_detector):
        """

        :param base_clf: prototype for the patch classifier
        :param classes: classes on which the patch classifier gets trained on
        :param base_drift_detector: Prototype for the class based BD3 instance
                                    Checks whether the patch is suitable for some new input data
        """

        self._clf = BaseClassifier(base_clf)
        self._classes = classes
        self._drift_detector = base_drift_detector.clone()

    def fit(self, train_data, validation_data):
        """

        :param train_data: tuple(data, labels), training data for the patch classifier
        :param validation_data: tuple(data, labels), validation data for the patch classifier
        :return:
        """

        # fit the patch classifier on the training data
        self._clf.partial_fit(train_data[0], train_data[1], classes=self._classes, validation_data=validation_data)

        # update the patches concept
        self._update_concept(validation_data)

    def predict(self, data):
        return self._clf.predict(data)

    def _update_concept(self, valdiation_data):
        # validate patch classifier on the validation data
        prediction = self._clf.predict(valdiation_data[0])

        # update the patch concept modeled by the drift detector based on the validation result
        self._drift_detector.add_element(prediction, valdiation_data[1], classifier_changed=True)

    def check_concept_similarity(self, data, labels):
        """ Checks whether the patch is suitable for some new input data

        :param data: test_data
        :param labels: test_labels
        :return: list, Classes in the test data that can not be covered by the patch
        """

        # predict the test data
        prediction = self.predict(data)

        # add the test data to the drift detector without changing the distribution
        self._drift_detector.add_element(prediction, labels, classifier_changed=False)

        # check whether the drift detector detects differing classes or not
        differing_classes = self._drift_detector.drifting_classes
        return differing_classes

    @property
    def classes(self):
        return self._classes
