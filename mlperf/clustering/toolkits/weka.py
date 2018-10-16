"""Weka clustering"""

__author__ = "Vincenzo Musco (http://www.vmusco.com)"

import subprocess
from os import path
from mlperf.clustering import clusteringtoolkit
from mlperf.tools.config import JAVA_EXE
from mlperf.tools.static import WEKA_ALGO, WEKA_UNORM_ALGO, JAVA_CLASSPATH


# Checking requirements on import
if not path.exists(JAVA_EXE):
    raise FileNotFoundError("Unable to locate a valid JAVA installation folder")

class Weka(clusteringtoolkit.ClusteringToolkit):
    """
    Default (normalized for kmeans at least) version of Weka
    """
    def __init__(self, normalized=True):
        super().__init__()
        self.normalized = normalized

    def toolkit_name(self):
        return WEKA_ALGO

    # https://stackoverflow.com/questions/6685961/weka-simple-k-means-clustering-assignments
    def run_kmeans_plus_plus(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None):
        output_file, centroids_file = self._prepare_files(dataset_name, run_info, True)
        weka_command = "WekaRun" if self.normalized else "WekaRunNorm"
        command_parts = [JAVA_EXE, "-classpath", JAVA_CLASSPATH, weka_command, src_file, output_file, centroids_file]
        subprocess.call(command_parts)

    def run_hierarchical(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None):
        output_file = self._prepare_files(dataset_name, run_info, False)
        command_parts = [JAVA_EXE, "-Xmx100g", "-classpath", JAVA_CLASSPATH, "HierarchicalWekaRun", src_file, output_file]
        subprocess.call(command_parts)

    def run_gaussian(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None):
        output_file = self._prepare_files(dataset_name, run_info, False)
        command_parts = [JAVA_EXE, "-Xmx100g", "-classpath", JAVA_CLASSPATH, "EMWekaRun", src_file, output_file]
        subprocess.call(command_parts)


class WekaUnorm(Weka):
    """
    Not normalized version of Weka
    """
    def __init__(self):
        super().__init__(True)

    def toolkit_name(self):
        return WEKA_UNORM_ALGO

    # https://stackoverflow.com/questions/6685961/weka-simple-k-means-clustering-assignments
    def run_kmeans_plus_plus(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None):
        output_file, centroids_file = self._prepare_files(dataset_name, run_info, True)
        weka_command = "WekaRun" if self.normalized else "WekaRunNorm"
        command_parts = [JAVA_EXE, "-classpath", JAVA_CLASSPATH, weka_command, src_file, output_file, centroids_file]
        subprocess.call(command_parts)

    @NotImplementedError
    def run_hierarchical(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None):
        pass

    @NotImplementedError
    def run_gaussian(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None):
        pass