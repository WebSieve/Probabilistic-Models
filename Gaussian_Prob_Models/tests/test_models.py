"""
Unit tests for datasets, LDA, and QDA (stdlib unittest only).
"""

import unittest

import numpy as np

from gaussian_prob_models import LDA, QDA, generate_synthetic_classification_data
from gaussian_prob_models.evaluate import evaluate_model, train_test_split


class TestDataGeneration(unittest.TestCase):
    def test_unequal_cov_shape_and_classes(self):
        X, y = generate_synthetic_classification_data(
            n_samples=400, dataset_type="unequal_cov", random_state=42
        )
        self.assertEqual(X.shape, (400, 2))
        self.assertEqual(y.shape, (400,))
        self.assertEqual(sorted(np.unique(y).tolist()), [0, 1])

    def test_equal_cov_shape_and_classes(self):
        X, y = generate_synthetic_classification_data(
            n_samples=400, dataset_type="equal_cov", random_state=42
        )
        self.assertEqual(X.shape, (400, 2))
        self.assertEqual(sorted(np.unique(y).tolist()), [0, 1])

    def test_multiclass_has_three_classes(self):
        X, y = generate_synthetic_classification_data(
            n_samples=399, dataset_type="multiclass", random_state=42
        )
        self.assertEqual(X.shape[1], 2)
        self.assertEqual(sorted(np.unique(y).tolist()), [0, 1, 2])

    def test_high_dim_respects_n_features(self):
        X, y = generate_synthetic_classification_data(
            n_samples=100, dataset_type="high_dim", n_features=5, random_state=42
        )
        self.assertEqual(X.shape, (100, 5))
        self.assertEqual(sorted(np.unique(y).tolist()), [0, 1])

    def test_reproducible_with_same_seed(self):
        X1, y1 = generate_synthetic_classification_data(random_state=7)
        X2, y2 = generate_synthetic_classification_data(random_state=7)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_unknown_dataset_type_raises(self):
        with self.assertRaises(ValueError):
            generate_synthetic_classification_data(dataset_type="nope")


class TestQDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = generate_synthetic_classification_data(
            n_samples=600, dataset_type="unequal_cov", random_state=42
        )
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(X, y)
        cls.model = QDA()
        cls.model.fit(cls.X_train, cls.y_train)

    def test_predict_shape(self):
        preds = self.model.predict(self.X_test)
        self.assertEqual(preds.shape, (len(self.y_test),))

    def test_predict_only_known_classes(self):
        preds = self.model.predict(self.X_test)
        self.assertTrue(set(np.unique(preds)) <= set(self.model.classes))

    def test_proba_rows_sum_to_one(self):
        proba = self.model.predict_proba(self.X_test)
        self.assertEqual(proba.shape, (len(self.y_test), 2))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)

    def test_log_proba_shape(self):
        ll = self.model.predict_log_proba(self.X_test)
        self.assertEqual(ll.shape, (len(self.y_test), 2))

    def test_per_class_covariances_differ(self):
        # QDA's whole point: one covariance per class
        self.assertFalse(
            np.allclose(self.model.cov[0], self.model.cov[1]),
            "class covariances should differ on unequal_cov data",
        )

    def test_accuracy_on_showcase_data(self):
        acc = (self.model.predict(self.X_test) == self.y_test).mean()
        self.assertGreater(acc, 0.85)


class TestLDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = generate_synthetic_classification_data(
            n_samples=600, dataset_type="equal_cov", random_state=42
        )
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(X, y)
        cls.model = LDA()
        cls.model.fit(cls.X_train, cls.y_train)

    def test_predict_shape(self):
        self.assertEqual(self.model.predict(self.X_test).shape, (len(self.y_test),))

    def test_proba_rows_sum_to_one(self):
        proba = self.model.predict_proba(self.X_test)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)

    def test_pooled_covariance_shared_across_classes(self):
        # LDA's whole point: a single pooled covariance reused per class
        self.assertEqual(self.model.cov.shape, (2, 2, 2))
        np.testing.assert_array_equal(self.model.cov[0], self.model.cov[1])

    def test_accuracy_on_showcase_data(self):
        acc = (self.model.predict(self.X_test) == self.y_test).mean()
        self.assertGreater(acc, 0.85)


class TestMulticlass(unittest.TestCase):
    def test_both_models_support_three_classes(self):
        X, y = generate_synthetic_classification_data(
            n_samples=399, dataset_type="multiclass", random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        for cls in (LDA, QDA):
            with self.subTest(model=cls.__name__):
                ev = evaluate_model(cls, X_train, y_train, X_test, y_test)
                self.assertEqual(ev["model"].predict(X_test).shape, (len(y_test),))
                self.assertGreater(ev["test_acc"], 0.5)


if __name__ == "__main__":
    unittest.main()
