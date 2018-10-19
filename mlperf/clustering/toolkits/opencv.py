"""OpenCV clustering"""

__author__ = "Vincenzo Musco (http://www.vmusco.com)"

import cv2
import numpy as np

from mlperf.clustering import clusteringtoolkit
from mlperf.clustering.clusteringtoolkit import ClusteringToolkit
from mlperf.tools.static import OPENCV_ALGO


class OpenCV(clusteringtoolkit.ClusteringToolkit):
    def __init__(self):
        """
        OpenCV exposes any seed parameter
        """
        super().__init__(None)

    def toolkit_name(self):
        return OPENCV_ALGO

    @staticmethod
    def _clustering_to_list(data_without_target, clusters):
        clusters_list = []
        for index, row in data_without_target.iterrows():
            clusters_list.append([index, clusters[index].item(0)])
        return clusters_list

    @staticmethod
    def _centroids_to_list(centers):
        centers_list = []
        for row in centers:
            centers_list.append(row.tolist())
        return centers_list

    @staticmethod
    def _save_run(ret, data_without_target, output_file, centroids_file):
        ClusteringToolkit._save_clustering(OpenCV._clustering_to_list(data_without_target, ret[1]), output_file)
        ClusteringToolkit._save_centroids(OpenCV._centroids_to_list(ret[2]), centroids_file)

    def run_kmeans_plus_plus(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None,
                             nb_iterations=None):
        output_file, centroids_file = self._prepare_files(dataset_name, run_info, True)

        # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100 if nb_iterations is None else nb_iterations,
                    0.0)

        ret = cv2.kmeans(np.float32(data_without_target.values), nb_clusters, None, criteria, 10,
                         flags=cv2.KMEANS_PP_CENTERS)
        OpenCV._save_run(ret, data_without_target, output_file, centroids_file)

        return output_file, {"centroids": centroids_file}

    def run_kmeans(self, nb_clusters, src_file, data_without_target, dataset_name, initial_clusters_file,
                   initial_clusters, run_number, run_info=None, nb_iterations=None):
        output_file, centroids_file = self._prepare_files(dataset_name, run_info, True)

        # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100 if nb_iterations is None else nb_iterations,
                    0.0)

        ret = cv2.kmeans(np.float32(data_without_target.values), nb_clusters, initial_clusters, criteria, 10,
                         flags=cv2.KMEANS_USE_INITIAL_LABELS)
        OpenCV._save_run(ret, data_without_target, output_file, centroids_file)

        return output_file, {"centroids": centroids_file}
