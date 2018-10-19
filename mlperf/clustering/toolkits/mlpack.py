"""MLPack clustering"""

__author__ = "Vincenzo Musco (http://www.vmusco.com)"

import csv
import time
from os import unlink, path
import subprocess

from mlperf.clustering import clusteringtoolkit
from mlperf.clustering.clusteringtoolkit import ClusteringToolkit
from mlperf.tools.config import TEMPFOLDER, MLPACK_BIN
from mlperf.tools.static import MLPACK_ALGO

# Checking requirements on import
binaries = ["{}/mlpack_kmeans".format(MLPACK_BIN), "{}/mlpack_mean_shift".format(MLPACK_BIN)]
for binary in binaries:
    if not path.exists(binary):
        raise FileNotFoundError("Unable to locate mlpack installation directory")


class MLPack(clusteringtoolkit.ClusteringToolkit):
    def toolkit_name(self):
        return MLPACK_ALGO

    @staticmethod
    def _save_mlpack_output(input_file, output_file):
        with open(input_file, 'r') as csv_file:
            with open(output_file, 'w') as resultFile:
                i = 0
                resultReader = csv.reader(csv_file)
                for row in resultReader:
                    resultFile.write("{},{}\n".format(i, int(float(row[-1]))))
                    i += 1

    # http://mlpack.org/man/kmeans.html
    # http://www.mlpack.org/docs/mlpack-2.1.0/doxygen/formatdoc.html
    def run_kmeans(self, nb_clusters, src_file, data_without_target, dataset_name, initial_clusters_file,
                   initial_clusters, run_number, run_info=None, nb_iterations=None):
        output_file, centroids_file = self._prepare_files(dataset_name, run_info, True)
        temp_file = self._dump_data_on_clean_csv(data_without_target)
        temp_file2 = "{}/{}b.csv".format(TEMPFOLDER, int(time.time()))

        # -a naive = Lloyd
        command_parts = ["{}/mlpack_kmeans".format(MLPACK_BIN), "--clusters", str(nb_clusters), "-i", temp_file, "-I",
                         initial_clusters_file, "-a", "naive", "-o", temp_file2, "-C", centroids_file]

        if nb_iterations is not None:
            command_parts.extend(['-m', '{}'.format(nb_iterations)])
        if self.seed is not None:
            command_parts.extend(['-s', '{}'.format(self.seed)])

        subprocess.call(command_parts)

        self._save_mlpack_output(temp_file2, output_file)
        unlink(temp_file)
        unlink(temp_file2)

        return output_file, {"centroids": centroids_file}

    def run_kmeans_plus_plus(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None,
                             nb_iterations=None):
        output_file, centroids_file = self._prepare_files(dataset_name, run_info, True)
        temp_file = self._dump_data_on_clean_csv(data_without_target)
        temp_file2 = "{}/{}b.csv".format(TEMPFOLDER, int(time.time()))

        command_parts = ["{}/mlpack_kmeans".format(MLPACK_BIN), "--clusters", str(nb_clusters), "-i", temp_file, "-o",
                         temp_file2, "-a", "naive", "-C", centroids_file]

        if nb_iterations is not None:
            command_parts.extend(['-m', '{}'.format(nb_iterations)])
        if self.seed is not None:
            command_parts.extend(['-s', '{}'.format(self.seed)])

        subprocess.call(command_parts)

        self._save_mlpack_output(temp_file2, output_file)
        unlink(temp_file)
        unlink(temp_file2)

        return output_file, {"centroids": centroids_file}

    def run_meanshift(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None):
        output_file = self._prepare_files(dataset_name, run_info, False)

        temp_file = ClusteringToolkit._dump_data_on_clean_csv(data_without_target)
        temp_file2 = ClusteringToolkit.create_temporary_file()

        command_parts = ["{}/mlpack_mean_shift".format(MLPACK_BIN), "-i", temp_file, "-o", temp_file2]
        subprocess.call(command_parts)

        self._save_mlpack_output(temp_file2, output_file)

        unlink(temp_file)
        unlink(temp_file2)

        return output_file, {}
