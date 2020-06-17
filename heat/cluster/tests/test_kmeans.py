import os
import unittest

import heat as ht

from ...core.tests.test_suites.basic_test import TestCase


class TestKMeans(TestCase):
    def test_clusterer(self):
        kmeans = ht.cluster.KMeans()
        self.assertTrue(ht.is_estimator(kmeans))
        self.assertTrue(ht.is_clusterer(kmeans))

    def test_get_and_set_params(self):
        kmeans = ht.cluster.KMeans()
        params = kmeans.get_params()

        self.assertEqual(
            params,
            {"n_clusters": 8, "init": "random", "max_iter": 300, "tol": 1e-4, "random_state": None},
        )

        params["n_clusters"] = 10
        kmeans.set_params(**params)
        self.assertEqual(10, kmeans.n_clusters)

    def test_fit_iris_unsplit(self):
        for split in [None, 0]:
            # get some test data
            iris = ht.load("heat/datasets/data/iris.csv", sep=";", split=split)

            # fit the clusters
            k = 3
            kmeans = ht.cluster.KMeans(n_clusters=k)
            kmeans.fit(iris)

            # check whether the results are correct
            self.assertIsInstance(kmeans.cluster_centers_, ht.DNDarray)
            self.assertEqual(kmeans.cluster_centers_.shape, (k, iris.shape[1]))
            # same test with init=kmeans++
            kmeans = ht.cluster.KMeans(n_clusters=k, init="kmeans++")
            kmeans.fit(iris)

            # check whether the results are correct
            self.assertIsInstance(kmeans.cluster_centers_, ht.DNDarray)
            self.assertEqual(kmeans.cluster_centers_.shape, (k, iris.shape[1]))

    def test_exceptions(self):
        # get some test data
        iris_split = ht.load("heat/datasets/data/iris.csv", sep=";", split=1)

        # build a clusterer
        k = 3
        kmeans = ht.cluster.KMeans(n_clusters=k)

        with self.assertRaises(NotImplementedError):
            kmeans.fit(iris_split)
        with self.assertRaises(ValueError):
            kmeans.set_params(foo="bar")
        with self.assertRaises(ValueError):
            kmeans = ht.cluster.KMeans(n_clusters=k, init="random_number")
            kmeans.fit(iris_split)
