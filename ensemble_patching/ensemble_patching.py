import numpy as np
from skmultiflow.core.base import StreamModel
from core.classifiers.base_classifier import BaseClassifier
from sklearn.model_selection import train_test_split
from ensemble_patching.patch import Patch
from ensemble_patching.conductor import Conductor
from sklearn.metrics import accuracy_score
from core.classifiers.mlp_classifier import MLPClassifier
from beta_distribution_drift_detector.bdddc import BDDDC
from sklearn.ensemble import RandomForestClassifier
from core.classifiers.gru_classifier import GRUClassifier
from core.dim_reduction.ae_dim_reductor import AEDimReductor
from core.dim_reduction.autoencoder import AutoEncoder


class EnsemblePatching(StreamModel):
    """ Ensemble Patching algorithm proposed in:
        Kauschke, Sebastian, and Fleckenstein, Lukas, and Johannes Fürnkranz:
        "Mending is Better than Ending: Adapting Immutable Classifiers to Nonstationary Environments using Ensembles
        of Patches", IJCNN 2019
    """

    def __init__(self, base_black_box_clf=RandomForestClassifier(n_estimators=100),
                 base_patch_clf=MLPClassifier([100]),
                 base_conductor=GRUClassifier(n_units=100),
                 black_box_bdddc=BDDDC(n_stable_assumed_batches=2),
                 base_patch_bdddc=BDDDC(n_stable_assumed_batches=0, decay_shape_a=0, decay_shape_b=0),
                 feature_embedding=AEDimReductor(AutoEncoder(encoding_layers=[100], encoding_activations=['linear'], dropout=0.2)),
                 pretrain_size=10000,
                 validation_split=0.2):

        """

        :param base_black_box_clf: prototype classifier for the black-box classifier
        :param base_patch_clf: prototype classifier for the patch classifiers
        :param base_conductor: prototype RNN classifier for the conductor
        :param black_box_bdddc: prototype class based BD3 applied to the black-box classifier
        :param base_patch_bdddc: prototype class based BD3 applied to the patches
        :param feature_embedding: either 'no_features', 'all_features', or a dimensionality reduction model
                                  Defines if/which data features are given to the conductor in addition to the
                                  ensemble members predictions
        :param pretrain_size: number of samples on which the black-box classifier gets trained
        :param validation_split: fraction of the batch size which is used as validation split for training patches
        """

        self._base_black_box_clf = base_black_box_clf
        self._base_patch_clf = base_patch_clf
        self._base_conductor = base_conductor
        self._black_box_bdddc = black_box_bdddc
        self._base_patch_bdddc = base_patch_bdddc
        self._feature_embedding = feature_embedding
        self._pretrain_size = pretrain_size
        self._validation_split = validation_split

        self._black_box_clf = BaseClassifier(self._base_black_box_clf)
        self._global_sample_count = 0

        self._patches = []
        self._conductor = None
        super().__init__()

    def partial_fit(self, X, y, classes=None, weight=None):
        """ Fits the algorithm to some input data

        :param X: np.array(n_samples, n_features), Training data
        :param y: np.array(n_samples), Training labels
        :param classes: np.array(n_classes) classes of the current environment
        :param weight: ignored
        :return:
        """

        # Init phase: Train the black-box classifier
        if self._global_sample_count < self._pretrain_size:
            self._black_box_clf.fit(X, y)
        else:
            # check for drift
            pred_black_box_clf = self._black_box_clf.predict(X)
            self._black_box_bdddc.add_element(pred_black_box_clf, y, classifier_changed=False)

            # fit ensemble if base concept drifted
            if self._black_box_bdddc.detected_change():
                drifting_classes = self._black_box_bdddc.drifting_classes
                self._fit_ensemble(X, y, drifting_classes)

            # fit feature embedding if base concept is stable
            else:
                if type(self._feature_embedding) is not str:
                    self._feature_embedding.partial_fit(X)

                # Update the conductor
                if self._conductor is not None:
                    self._fit_conductor(X, y)

        self._global_sample_count += len(X)

    def predict(self, X):
        """ Ensemble prediction given some input data

        :param X: np.array(n_samples, n_features), Input data
        :return: np.array(n_samples), Ensemble Prediction
        """

        # If no drift has been detected yet, return the prediction of the black-box classifier
        if self._conductor is None:
            return self._black_box_clf.predict(X)

        # If the conductor exists, return the conductor prediction
        else:
            conductor_sequence = self._preprocess_conductor_sequence(X)
            return self._conductor.predict(conductor_sequence)

    def score(self, X, y):
        """ Returns the models accuracy score

        :param X: np.array(n_samples, n_features), Input data
        :param y: np.array(n_samples), Ensemble Prediction
        :return: float, Accuracy
        """
        prediction = self.predict(X)
        accuracy = accuracy_score(y, prediction)
        return accuracy

    def _fit_ensemble(self, X, y, corrupted_classes):
        """ Fit the ensemble if some classes to not belong to the black-box classifiers concept
        """

        # Split the input data in train and validation set
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self._validation_split)

        # If no patches available, fit first patch
        if len(self._patches) == 0:
            self._fit_new_patch(train_data=(X_train, y_train),
                                val_data=(X_val, y_val),
                                corrupted_classes=corrupted_classes)

        else:
            # Check if existing patches can cover the drifting classes
            uncovered_classes, covered_classes = self._check_patch_covering(X, y, corrupted_classes)

            # Update patches on the new data if they are suitable
            if np.size(covered_classes) != 0:
                self._update_existing_patches(train_data=(X_train, y_train),
                                              val_data=(X_val, y_val),
                                              covered_classes=covered_classes)

            # Fit new patch if the existing ones can not cover all of the drifting classes
            if len(uncovered_classes) != 0:
                self._fit_new_patch(train_data=(X_train, y_train),
                                    val_data=(X_val, y_val),
                                    corrupted_classes=uncovered_classes)

        # Fit the conductor
        self._fit_conductor(X, y)

    def _fit_new_patch(self, train_data, val_data, corrupted_classes):
        """ Fits a new patch given some data and the classes that can not get covered by the existing ensemble
        """

        # Get sub data according to drifting classes
        patch_train_data = self._build_class_based_subset(train_data[0], train_data[1], corrupted_classes)
        patch_val_data = self._build_class_based_subset(val_data[0], val_data[1], corrupted_classes)

        # Create new patch and fit it on the subset
        patch = Patch(self._base_patch_clf, corrupted_classes, self._base_patch_bdddc)
        patch.fit(patch_train_data, patch_val_data)

        # Add patch to the ensemble
        self._patches.append(patch)

    def _fit_conductor(self, X, y):
        """ Fits the conductor
        """

        # Pre-process the conductor input data
        rnn_sequence = self._preprocess_conductor_sequence(X)

        # Build the conductor if the current drift is the first one
        if self._conductor is None:
            self._conductor = Conductor(self._base_conductor)

        # Fit the conductor
        self._conductor.fit(rnn_sequence, y)

    def _preprocess_conductor_sequence(self, data):
        """ Builds the Conductor input sequence consisting of the ensemble members predictions and optional data features
        """

        # Observe the prediction from the black-box classifier and all patches
        pred_black_box_clf = self._black_box_clf.predict(data)
        pred_patches = np.vstack([patch.predict(data) for patch in self._patches])

        # Stack all ensemble member predictions
        all_clf_predictions = np.vstack((pred_black_box_clf, np.vstack(pred_patches))).T

        # If no data features should be included, build sequence consisting only of member predictions
        if self._feature_embedding == 'no_features':
            features = [all_clf_predictions[:, i][:, np.newaxis]
                        for i in range(np.shape(all_clf_predictions)[1])]

        # If data features should be included, build sequence consisting of member predictions and data features
        else:
            data_features = self._get_rnn_data_features(data)
            features = [np.hstack((all_clf_predictions[:, i][:, np.newaxis], data_features))
                        for i in range(np.shape(all_clf_predictions)[1])]

        rnn_input_sequence = np.stack(features, axis=1)
        return rnn_input_sequence

    def _get_rnn_data_features(self, data):
        """ Returns the data features that get included in the Conductor sequence. Either the original data or some
            embedding.
        """

        # If the original data features should be included, return the data itself
        if self._feature_embedding == 'all_features':
            features = data

        # If a feature embedding should be applied, transform the input data in the embedding space
        else:
            features = self._feature_embedding.transform(data)
        return features

    def _check_patch_covering(self, X, y, corrupted_classes):
        """ Checks for each class on which the black-box classifiers errs if an existing patch can cover the drift
        """

        tmp_classes = np.unique(y)
        ensemble_covered_classes = []

        for patch in self._patches:
            # If patch has been trained on some of the error classes, check if the concept is the same
            if any(np.isin(patch.classes, tmp_classes)):
                patch_X, patch_y = self._build_class_based_subset(X, y, patch.classes)
                differing_classes = patch.check_concept_similarity(patch_X, patch_y)
                patch_covered_classes = patch.classes[np.argwhere(~np.isin(patch.classes, differing_classes)).flatten()]
            else:
                patch_covered_classes = np.array([])

            ensemble_covered_classes.append(patch_covered_classes)

        # Combine patch suitabilities
        all_covered_classes = np.unique(np.hstack(ensemble_covered_classes))
        uncoverd_classes = corrupted_classes[np.argwhere(~np.isin(corrupted_classes, all_covered_classes)).flatten()]
        return uncoverd_classes, ensemble_covered_classes

    def _update_existing_patches(self, train_data, val_data, covered_classes):
        """ Update patches on new data for which they are suitable
        """

        for p_idx, p_covered_classes in enumerate(covered_classes):
            if len(p_covered_classes) != 0:
                all_patch_classes = self._patches[p_idx].classes

                # update patch if all classes match
                if np.array_equal(p_covered_classes, all_patch_classes):
                    patch_X_train, patch_y_train = self._build_class_based_subset(train_data[0],
                                                                                  train_data[1],
                                                                                  p_covered_classes)

                    patch_X_val, patch_y_val = self._build_class_based_subset(val_data[0],
                                                                              val_data[1],
                                                                              p_covered_classes)

                    self._patches[p_idx].fit(train_data=(patch_X_train, patch_y_train),
                                             validation_data=(patch_X_val, patch_y_val))

    @staticmethod
    def _build_class_based_subset(X, y, classes):
        sample_idx = np.isin(y, classes)
        class_subset = X[sample_idx], y[sample_idx]
        return class_subset

    def get_info(self):
        """ Returns the models parameters as string

        :return:    string
                    The model's parameters
        """

        description = type(self).__name__ + ': '
        description += 'base_black_box_classifier - %s, ' % type(self._base_black_box_clf).__name__
        description += 'base_patch_classifier - %s, ' % type(self._base_patch_clf).__name__
        description += 'base_conductor - %s, ' % type(self._base_conductor).__name__
        description += 'drift_detector - %s, ' % type(self._black_box_bdddc).__name__

        if type(self._feature_embedding) is str:
            description += 'feature_embedding - %s, ' % self._feature_embedding
        else:
            description += 'feature_embedding - %s, ' % type(self._feature_embedding).__name__

        description += 'pretrain_size - %s, ' % str(self._pretrain_size)
        return description

    def reset(self):
        """ Resets the model
        """
        self._black_box_clf = BaseClassifier(self._base_black_box_clf)
        self._black_box_bdddc.reset()
        self._global_sample_count = 0

        self._patches = []
        self._conductor = None

        if type(self._feature_embedding) is not str:
            self._feature_embedding.reset()

    def fit(self, X, y, classes=None, weight=None):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError


