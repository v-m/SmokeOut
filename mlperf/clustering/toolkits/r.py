"""R clustering"""

__author__ = "Vincenzo Musco (http://www.vmusco.com)"

from os import path
import subprocess

from mlperf.clustering import clusteringtoolkit
from mlperf.tools.config import R_BIN
from mlperf.tools.static import R_ALGO, R_SCRIPT_BASE_DIR
import pandas

# Checking requirements on import
if not path.exists(R_SCRIPT_BASE_DIR):
    raise FileNotFoundError("Unable to locate matlab installation directory")

class R(clusteringtoolkit.ClusteringToolkit):
    def toolkit_name(self):
        return R_ALGO

    def run_kmeans(self, nb_clusters, src_file, data_without_target, dataset_name, initial_clusters_file,
                   initial_clusters, run_number, run_info=None, nb_iterations=None):
        # if initialClustersCsvFile is None and hundredIters:
        #     print("R kcca function don't have the iteration parameter !")
        #     return

        output_file, centroids_file = self._prepare_files(dataset_name, run_info, True)

        script_file = "kmeans_test_init_clusters.R" if nb_iterations is None else "kmeans_test_init_clusters_n_it.R"
        command_parts = [R_BIN, "--no-save", "--quiet", path.join(R_SCRIPT_BASE_DIR, script_file),
                         src_file, output_file, centroids_file, initial_clusters_file]
        if nb_iterations is not None:
            command_parts.append("{}".format(nb_iterations))

        subprocess.call(command_parts)

        dta = pandas.read_csv(centroids_file)
        dta.drop(dta.columns[[0]], axis=1).to_csv(centroids_file, index=False, header=False)

    def run_kmeans_plus_plus(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None,
                             nb_iterations=None):
        output_file, centroids_file = self._prepare_files(dataset_name, run_info, True)

        script_file = "kmeans_test.R" if nb_iterations is None else "kmeans_test_n_it.R"
        command_parts = [R_BIN, "--no-save", "--quiet", path.join(R_SCRIPT_BASE_DIR, script_file), src_file,
                         output_file, centroids_file]
        if nb_iterations is not None:
            command_parts.append("{}".format(nb_iterations))

        subprocess.call(command_parts)

        dta = pandas.read_csv(centroids_file)
        dta.drop(dta.columns[[0]], axis=1).to_csv(centroids_file, index=False, header=False)

    # TODO 100iter?

    def run_hierarchical(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None):
        output_file = self._prepare_files(dataset_name, run_info, False)

        command_parts = [R_BIN, "--no-save", "--quiet", path.join(R_SCRIPT_BASE_DIR, "hierarchical_test.R"), src_file,
                         output_file]
        subprocess.call(command_parts)
