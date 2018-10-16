"""Tensorflow clustering"""

__author__ = "Vincenzo Musco (http://www.vmusco.com)"

import tensorflow as tf
from numpy.core.tests.test_mem_overlap import xrange
from tensorflow.python.framework import constant_op

from mlperf.clustering import clusteringtoolkit
from mlperf.clustering.clusteringtoolkit import ClusteringToolkit
from mlperf.tools.static import TENSORFLOW_ALGO


class TensorFlow(clusteringtoolkit.ClusteringToolkit):
    def toolkit_name(self):
        return TENSORFLOW_ALGO

    @staticmethod
    def _train_kpp(input_fn, kmeans, iterations = 10):
        # train
        num_iterations = 10
        previous_centers = None
        for _ in xrange(num_iterations):
            kmeans.train(input_fn)

    @staticmethod
    def _build_points_and_input_fn(data_without_target):
        points = data_without_target.values

        def input_fn():
            return tf.train.limit_epochs(tf.convert_to_tensor(points, dtype=tf.float32), num_epochs=1)

        return points, input_fn


    @staticmethod
    def _clustering_to_list(points, cluster_indices):
        clusters_list = []
        for index, point in enumerate(points):
            cluster_index = cluster_indices[index]
            clusters_list.append([index, cluster_index])
        return clusters_list

    @staticmethod
    def _centroids_to_list(model):
        centers_list = []
        for row in model.cluster_centers():
            centers_list.append(row.tolist())
        return centers_list

    # https://www.tensorflow.org/api_docs/python/tf/contrib/factorization/KMeansClustering
    def run_kmeans(self, nb_clusters, src_file, data_without_target, dataset_name, initial_clusters_file,
                   initial_clusters, run_number, run_info=None):
        output_file, centroids_file = self._prepare_files(dataset_name, run_info, True)

        kmeans = tf.contrib.factorization.KMeansClustering(num_clusters=nb_clusters,
                                                           initial_clusters=initial_clusters, use_mini_batch=False)

        points, input_fn = TensorFlow._build_points_and_input_fn(data_without_target)
        TensorFlow._train_kpp(input_fn, kmeans, 10)
        cluster_indices = list(kmeans.predict_cluster_index(input_fn))
        ClusteringToolkit._save_clustering(TensorFlow._clustering_to_list(points, cluster_indices), output_file)
        ClusteringToolkit._save_centroids(TensorFlow._centroids_to_list(kmeans), centroids_file)


    def run_kmeans_plus_plus(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None):
        output_file, centroids_file = self._prepare_files(dataset_name, run_info, True)

        kmeans = tf.contrib.factorization.KMeansClustering(num_clusters=nb_clusters, use_mini_batch=False)

        points, input_fn = TensorFlow._build_points_and_input_fn(data_without_target)
        TensorFlow._train_kpp(input_fn, kmeans, 10)
        cluster_indices = list(kmeans.predict_cluster_index(input_fn))
        ClusteringToolkit._save_clustering(TensorFlow._clustering_to_list(points, cluster_indices), output_file)
        ClusteringToolkit._save_centroids(TensorFlow._centroids_to_list(kmeans), centroids_file)

    def run_gaussian(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None):
        output_file, centroids_file = self._prepare_files(dataset_name, run_info, True)
        points = data_without_target.values

        def get_input_fn():
            def input_fn():
                return constant_op.constant(points.astype(np.float32)), None

            return input_fn

        gmm = tf.contrib.factorization.GMM(num_clusters=nb_clusters)
        gmm.fit(input_fn=get_input_fn(), steps=1)

        cluster_indices = list(gmm.predict_assignments())
        ClusteringToolkit._save_clustering(TensorFlow._clustering_to_list(points, cluster_indices), output_file)
        ClusteringToolkit._save_centroids(TensorFlow._centroids_to_list(gmm), centroids_file)

    def run_gaussian_initial_starting_points(self, nb_clusters, src_file, data_without_target, dataset_name,
                                            initial_clusters_file, initial_clusters, run_number, run_info=None):
        output_file, centroids_file = self._prepare_files(dataset_name, run_info, True)
        points = data_without_target.values

        def get_input_fn():
            def input_fn():
                return constant_op.constant(points.astype(np.float32)), None

            return input_fn

        gmm = tf.contrib.factorization.GMM(num_clusters=nb_clusters, initial_clusters=initial_clusters)
        gmm.fit(input_fn=get_input_fn(), steps=1)

        cluster_indices = list(gmm.predict_assignments())
        ClusteringToolkit._save_clustering(TensorFlow._clustering_to_list(points, cluster_indices), output_file)
        ClusteringToolkit._save_centroids(TensorFlow._centroids_to_list(gmm), centroids_file)