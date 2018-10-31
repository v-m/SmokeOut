"""Weka clustering"""

__author__ = "Vincenzo Musco (http://www.vmusco.com)"

import subprocess
from os import path
from mlperf.clustering import clusteringtoolkit
from mlperf.tools.config import JAVA_EXE
from mlperf.tools.static import WEKA_TOOLKIT, WEKA_UNORM_TOOLKIT, JAVA_CLASSPATH


class Weka(clusteringtoolkit.ClusteringToolkit):
    """
    Default (normalized for kmeans at least) version of Weka
    """

    def __init__(self, normalized=True):
        super().__init__()
        self.normalized = normalized

    def check_toolkit_requirements(self):
        if not path.exists(JAVA_EXE):
            raise FileNotFoundError("Unable to locate a valid JAVA installation folder")

    def toolkit_name(self):
        return WEKA_TOOLKIT

    def run_kmeans_base(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None,
                        nb_iterations=None, mode=None):
        output_file, centroids_file = self._prepare_files(dataset_name, run_info, True)
        weka_rest = []

        if not self.normalized:
            weka_rest.append("unorm=1")

        if nb_iterations is not None:
            weka_rest.append("nbiter={}".format(nb_iterations))

        if self.seed is not None:
            weka_rest.append("seed={}".format(self.seed))

        if mode is not None:
            weka_rest.append("mode={}".format(mode))

        command_parts = [JAVA_EXE, "-classpath", JAVA_CLASSPATH, "WekaRun", src_file, output_file, centroids_file]

        if len(weka_rest) > 0:
            command_parts.append(";".join(weka_rest))

        subprocess.call(command_parts)
        return output_file, {"centroids": centroids_file}

    # https://stackoverflow.com/questions/6685961/weka-simple-k-means-clustering-assignments
    def run_kmeans_plus_plus(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None,
                             nb_iterations=None):
        return self.run_kmeans_base(nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info,
                                    nb_iterations, "kpp")

    def run_kmeans_random(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None,
                          nb_iterations=None):
        return self.run_kmeans_base(nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info,
                                    nb_iterations, "auto")

    def run_kmeans_auto(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None,
                        nb_iterations=None):
        return self.run_kmeans_base(nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info,
                                    nb_iterations, None)

    def run_hierarchical(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None):
        output_file, = self._prepare_files(dataset_name, run_info, False)
        # No seed or parameters for hierarchical
        command_parts = [JAVA_EXE, "-Xmx100g", "-classpath", JAVA_CLASSPATH, "HierarchicalWekaRun", src_file,
                         output_file]
        subprocess.call(command_parts)

        return output_file, {}

    def run_gaussian(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None):
        output_file, = self._prepare_files(dataset_name, run_info, False)

        weka_rest = []
        if self.seed is not None:
            weka_rest.append("seed={}".format(self.seed))

        command_parts = [JAVA_EXE, "-Xmx100g", "-classpath", JAVA_CLASSPATH, "EMWekaRun", src_file, output_file]

        if len(weka_rest) > 0:
            command_parts.append(";".join(weka_rest))

        subprocess.call(command_parts)

        return output_file, {}


class WekaUnorm(Weka):
    """
    Not normalized version of Weka
    """

    def __init__(self):
        super().__init__(False)

    def toolkit_name(self):
        return WEKA_UNORM_TOOLKIT

    @NotImplementedError
    def run_hierarchical(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None):
        pass

    @NotImplementedError
    def run_gaussian(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None):
        pass
