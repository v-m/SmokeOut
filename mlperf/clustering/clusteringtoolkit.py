"""Base implementation for a toolkit"""

__author__ = "Vincenzo Musco (http://www.vmusco.com)"
__date__ = "Oct 16, 2018"

import csv
import os
import random
import time

from mlperf.tools.config import TEMPFOLDER


class AlreadyRanException(BaseException):
    """Exception thrown if a specific run X is already computed (and thus skipped)"""
    pass


class ClusteringToolkit:
    """This class contains all functions and needed abstract methods to run a toolkit
    For adding a new algorithm, a new run_[algo_name](...) method should be added"""

    def __init__(self, seed=None):
        self.overwrite_ran_iterations = False
        self.seed = seed
        self.redirect_output_files = None
        self.debug = False
        self.temp_files = []

    def set_overwrite_ran_iterations(self, new_value):
        """Set to true to not skip already ran clusterings"""
        self.overwrite_ran_iterations = new_value

    def set_redirect_output_path(self, new_path):
        self.redirect_output_files = new_path

    def set_seed(self, new_seed):
        """Set the specified seed for next executions"""
        self.seed = new_seed

    @NotImplementedError
    def toolkit_name(self):
        pass

    def _dataset_out_file_name(self, dataset_name, ext="csv", run_info=None):
        return ClusteringToolkit.dataset_out_file_name_static(dataset_name, self.toolkit_name(), ext, run_info)

    def _centroid_out_file_name(self, dataset_name, ext="csv", run_info=None):
        return ClusteringToolkit.dataset_out_file_name_static(dataset_name, ClusteringToolkit._centroid_filename_for(self.toolkit_name()),
                                                              ext, run_info)

    def _prepare_files(self, dataset_name, run_info, centroids=False):
        dataset_out_name = dataset_name
        if self.redirect_output_files is not None:
            base_name = os.path.basename(dataset_name)
            dataset_out_name = os.path.join(self.redirect_output_files, base_name)

        output_file = self._dataset_out_file_name(dataset_out_name, run_info=run_info)
        centroids_file = self._centroid_out_file_name(dataset_out_name, run_info=run_info)

        if not self.overwrite_ran_iterations:
            if os.path.exists(output_file) and (not centroids or os.path.exists(centroids_file)):
                raise AlreadyRanException

        ret = [output_file]
        ret.extend([centroids_file] if centroids else [])
        return ret

    @staticmethod
    def _save_clustering(clustering, output_file):
        """
        Save the result clustering list of list with [index, cluster] entries to a CSV file
        :param clustering: the clustering result
        :param output_file: the filename
        """
        with open(output_file, 'w') as csv_file:
            file_writer = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL)

            for row in clustering:
                file_writer.writerow(row)

    @staticmethod
    def _save_centroids(centroids, output_file):
        """
        Save the result centroids list with [center, center, ...] entries to a CSV file
        :param clustering: the centroids result
        :param output_file: the filename
        """
        with open(output_file, 'w') as csv_file:
            file_writer = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL)

            for row in centroids:
                file_writer.writerow(row)

    @staticmethod
    def dataset_out_file_name_static(dataset_name, toolkit_name, ext="csv", run_info=None):
        """
        :param dataset_name: Dataset name
        :param toolkit_name: Toolkit name
        :param ext: File extension
        :param run_info: Optional informations
        :return: This method build an appropriate filename taking into consideration the current configuration
        """
        info_text = ".{}".format(run_info) if run_info is not None else ""
        return "{}.{}{}.{}".format(dataset_name, toolkit_name, info_text, ext)

    @staticmethod
    def _centroid_filename_for(algo_name):
        return "{}.centroids".format(algo_name)

    def _dump_data_on_clean_csv(self, data_without_target):
        temp_file = self.create_temporary_file()

        fp = open(temp_file, "w")
        fp.write(data_without_target.to_csv(index=False, header=False))
        fp.close()

        return temp_file

    def create_temporary_file(self):
        ret = "{}/{}_{}.csv".format(TEMPFOLDER, int(time.time()), random.randint(1, 10000))
        self.temp_files.append(ret)
        return ret

    def clean_temporary_files(self):
        for temp_file in self.temp_files:
            os.unlink(temp_file)

    @NotImplementedError
    def run_kmeans(self, nb_clusters, src_file, data_without_target, dataset_name, initial_clusters_file,
                   initial_clusters, run_number, run_info=None, nb_iterations=None):
        pass

    @NotImplementedError
    def run_kmeans_plus_plus(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None,
                             nb_iterations=None):
        pass

    @NotImplementedError
    def run_hierarchical(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None):
        pass

    @NotImplementedError
    def run_spectral(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None):
        pass

    @NotImplementedError
    def run_ap(self, data_without_target, src_file, dataset_name, run_number, run_info=None):
        pass

    @NotImplementedError
    def run_dbscan(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None):
        pass

    @NotImplementedError
    def run_gaussian(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None):
        pass

    @NotImplementedError
    def run_meanshift(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None):
        pass

    @NotImplementedError
    def run_gaussian_initial_starting_points(self, nb_clusters, src_file, data_without_target, dataset_name,
                                            initial_clusters_file, initial_clusters, run_number, run_info=None):
        pass
