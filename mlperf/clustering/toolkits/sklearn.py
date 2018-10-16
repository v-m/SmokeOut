"""SKLearn clustering"""

__author__ = "Vincenzo Musco (http://www.vmusco.com)"

import inspect
from mlperf.clustering import clusteringtoolkit
from mlperf.clustering.clusteringtoolkit import ClusteringToolkit
from mlperf.tools.static import SKLEARN_ALGO, SKLEARN__FAST_ALGO, SKLEARN_TOL0_ALGO
import numpy as np
import sklearn.cluster

class Sklearn(clusteringtoolkit.ClusteringToolkit):
    """Toolkit class embedding utils functions"""
    def _clustering_to_list(self, data_without_target, model):
        clustering = []
        for index, row in data_without_target.iterrows():
            clustering.append([index, model[index]])
        return clustering

    def _centroids_to_list(self, model):
        centroids = []
        for row in model.cluster_centers_:
            centroids.append(row.tolist())
        return centroids


class SklearnCustomTolerance(Sklearn):
    """Sklearn with custom tolerance"""

    def __init__(self, tolerance):
        super().__init__()
        self.tolerance = tolerance

        if type(self.tolerance) == str:
            self.tolerance = float(self.tolerance)

    def toolkit_name(self):
        return SKLEARN_TOL0_ALGO

    def run_kmeans_plus_plus(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None):
        output_file, centroids_file = self._prepare_files(dataset_name, run_info, True)

        # Create a KMean model.
        sklearn_kmean_model = sklearn.cluster.KMeans(n_clusters=nb_clusters, tol=self.tolerance)
        sklearn_kmean_model.fit(data_without_target)
        sklearn_kmean_model.fit(data_without_target)

        ClusteringToolkit._save_clustering(self._clustering_to_list(data_without_target, sklearn_kmean_model.labels_), output_file)
        ClusteringToolkit._save_centroids(self._centroids_to_list(sklearn_kmean_model), centroids_file)

    def run_kmeans(self, nb_clusters, src_file, data_without_target, dataset_name, initial_clusters_file, initial_clusters, run_number, run_info=None):
        output_file, centroids_file = self._prepare_files(dataset_name, run_info, True)

        # Create a KMean model.
        sklearn_kmean_model = sklearn.cluster.KMeans(n_clusters=nb_clusters, init=initial_clusters, tol=self.tolerance)
        sklearn_kmean_model.fit(data_without_target)
        sklearn_kmean_model.fit(data_without_target)

        ClusteringToolkit._save_clustering(self._clustering_to_list(data_without_target, sklearn_kmean_model.labels_), output_file)
        ClusteringToolkit._save_centroids(self._centroids_to_list(sklearn_kmean_model), centroids_file)

    def run_gaussian(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None):
        output_file = self._prepare_files(dataset_name, run_info, False)

        built_model = sklearn.mixture.GaussianMixture(n_components=nb_clusters, tol=self.tolerance)
        built_model.fit(data_without_target)

        predicted_labels = built_model.predict(data_without_target)
        self._save_clustering(self._clustering_to_list(data_without_target, predicted_labels), output_file)


class SklearnVanilla(SklearnCustomTolerance):
    """Sklearn with default tolerance"""

    def toolkit_name(self):
        return SKLEARN_ALGO

    def __init__(self):
        super().__init__(0)
        self.setup_default_kmeans_tolerance()

    def setup_default_kmeans_tolerance(self):
        default_sklearn_tolerance = inspect.signature(sklearn.cluster.KMeans).parameters['tol']
        self.tolerance = default_sklearn_tolerance

    def setup_default_gaussian_tolerance(self):
        default_sklearn_tolerance = inspect.signature(sklearn.mixture.GaussianMixture).parameters['tol']
        self.tolerance = default_sklearn_tolerance

    def run_hierarchical(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None):
        output_file = self._prepare_files(dataset_name, run_info, False)

        built_model = sklearn.cluster.AgglomerativeClustering(n_clusters=nb_clusters)
        built_model.fit(data_without_target)

        self._save_clustering(self._clustering_to_list(data_without_target, built_model.labels_), output_file)

    def run_meanshift(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None):
        output_file, centroids_file = self._prepare_files(dataset_name, run_info, True)

        # Create model.
        model = sklearn.cluster.MeanShift()
        model.fit(data_without_target)

        self._save_clustering(self._clustering_to_list(data_without_target, model.labels_), output_file)
        ClusteringToolkit._save_centroids(self._centroids_to_list(model), centroids_file)

    def run_spectral(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None):
        output_file = self._prepare_files(dataset_name, run_info, False)

        check = False

        while not check:
            try:
                built_model = sklearn.cluster.SpectralClustering(n_clusters=nb_clusters)
                built_model.fit(data_without_target)
                check = True
            except np.linalg.linalg.LinAlgError:
                continue
            except AssertionError:
                continue

            self._save_clustering(self._clustering_to_list(data_without_target, built_model.labels_), output_file)

    def run_dbscan(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None):
        output_file = self._prepare_files(dataset_name, run_info, False)

        eps_value = 0.33 * run_number
        sample_value = run_number % 10
        if sample_value == 0:
            sample_value = 10
        built_model = sklearn.cluster.DBSCAN(eps=eps_value, min_samples=sample_value)
        built_model.fit(data_without_target)

        self._save_clustering(self._clustering_to_list(data_without_target, built_model.labels_), output_file)

    def run_ap(self, data_without_target, src_file, dataset_name, run_number, run_info=None):
        output_file = self._prepare_files(dataset_name, run_info, False)

        damping_value = 0.016 * run_number + 0.5
        built_model = sklearn.cluster.AffinityPropagation(damping=damping_value)
        built_model.fit(data_without_target)

        self._save_clustering(self._clustering_to_list(data_without_target, built_model.labels_), output_file)


class SklearnFast(Sklearn):
    def toolkit_name(self):
        return SKLEARN__FAST_ALGO
