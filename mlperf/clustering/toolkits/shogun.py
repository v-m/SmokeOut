"""Shogun clustering"""

__author__ = "Vincenzo Musco (http://www.vmusco.com)"

import shogun
import random
import numpy as np

from mlperf.clustering import clusteringtoolkit
from mlperf.clustering.clusteringtoolkit import ClusteringToolkit
from mlperf.tools.static import SHOGUN_ALGO


class Shogun(clusteringtoolkit.ClusteringToolkit):
    def toolkit_name(self):
        return SHOGUN_ALGO

    def _init(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

    @staticmethod
    def _clustering_to_list(data_without_target, model):
        clustering = []
        for index, row in data_without_target.iterrows():
            clustering.append([index, model[index].item(0)])
        return clustering

    @staticmethod
    def _centroids_to_list(model):
        centroids = []
        for row in model.transpose():
            centroids.append(row.tolist())
        return centroids

    @staticmethod
    def _kmeans_process(kmeans):
        # KMeans training
        kmeans.train()
        # cluster centers
        centers = kmeans.get_cluster_centers()
        # Labels for data points
        result = kmeans.apply()

        return centers, result

    def run_kmeans_plus_plus(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None,
                             nb_iterations=None):
        self._init()
        output_file, centroids_file = self._prepare_files(dataset_name, run_info, True)

        train_features = shogun.RealFeatures(data_without_target.values.astype("float64").transpose())
        # distance metric over feature matrix - Euclidean distance
        distance = shogun.EuclideanDistance(train_features, train_features)

        # KMeans object created
        kmeans = shogun.KMeans(nb_clusters, distance)
        # set KMeans++ flag
        kmeans.set_use_kmeanspp(True)

        if nb_iterations is not None:
            kmeans.set_max_iter(nb_iterations)

        centers, result = Shogun._kmeans_process(kmeans)
        ClusteringToolkit._save_clustering(Shogun._clustering_to_list(data_without_target, result), output_file)
        ClusteringToolkit._save_centroids(Shogun._centroids_to_list(centers), centroids_file)

        return output_file, {"centroids": centroids_file}

    def run_kmeans(self, nb_clusters, src_file, data_without_target, dataset_name, initial_clusters_file,
                   initial_clusters, run_number, run_info=None, nb_iterations=None):
        self._init()
        output_file, centroids_file = self._prepare_files(dataset_name, run_info, True)

        train_features = shogun.RealFeatures(data_without_target.values.astype("float64").transpose())
        # distance metric over feature matrix - Euclidean distance
        distance = shogun.EuclideanDistance(train_features, train_features)

        # KMeans object created
        kmeans = shogun.KMeans(nb_clusters, distance)
        # set new initial centers
        kmeans.set_initial_centers(initial_clusters.astype("float64").transpose())

        if nb_iterations is not None:
            kmeans.set_max_iter(nb_iterations)

        centers, result = Shogun._kmeans_process(kmeans)
        ClusteringToolkit._save_clustering(Shogun._clustering_to_list(data_without_target, result), output_file)
        ClusteringToolkit._save_centroids(Shogun._centroids_to_list(centers), centroids_file)

        return output_file, {"centroids": centroids_file}

    @NotImplementedError
    def run_hierarchical(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None):
        output_file, = self._prepare_files(dataset_name, run_info, False)

        train_features = shogun.RealFeatures(data_without_target.values.astype("float64").transpose())
        # distance metric over feature matrix - Euclidean distance
        distance = shogun.EuclideanDistance(train_features, train_features)

        hierarchical = shogun.Hierarchical(nb_clusters, distance)

        # TODO Makes the pyhon process dies!!!???!!!
        # d = hierarchical.get_merge_distances()
        # cp = hierarchical.get_cluster_pairs()
        # Toolkit._save_clustering(Shogun._clustering_to_list(data_without_target, result), output_file)

    @NotImplementedError
    def run_gaussian(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None):
        output_file, = self._prepare_files(dataset_name, run_info, False)

        train_features = shogun.RealFeatures(data_without_target.values.astype("float64").transpose())
        # distance metric over feature matrix - Euclidean distance
        # distance = shogun.EuclideanDistance(train_features, train_features)

        gmm = shogun.GMM(nb_clusters)
        gmm.set_features(train_features)
        gmm.train_em()

        print(gmm)
