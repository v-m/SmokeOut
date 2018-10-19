"""Matlab clustering"""

__author__ = "Vincenzo Musco (http://www.vmusco.com)"

import re
from os import unlink, path
import subprocess

from mlperf.clustering import clusteringtoolkit
from mlperf.clustering.clusteringtoolkit import ClusteringToolkit
from mlperf.tools.config import MATLAB_EXE, TEMPFOLDER
from mlperf.tools.static import MATLAB_ALGO

# Checking requirements on import
if not path.exists(MATLAB_EXE):
    raise FileNotFoundError("Unable to locate matlab installation directory")

class MatLab(clusteringtoolkit.ClusteringToolkit):
    def toolkit_name(self):
        return MATLAB_ALGO

    @staticmethod
    def _matlab_redirect_temp_folder(temp_dir):
        return "tempdir='{}';".format(temp_dir)

    @staticmethod
    def _parse_output(res, output_file, centroids_file):
        reading_index = True
        i = 0

        with open(output_file, 'w') as result_file:
            for line in res.decode().split("\n"):
                if reading_index and line == "===C===":
                    result_file.close()
                    result_file = open(centroids_file, 'w')
                    reading_index = False
                else:
                    if reading_index:
                        matches = re.fullmatch(" {5}([0-9]+)", line)

                        if matches is not None:
                            result_file.write("{},{}\n".format(i, matches.group(1)))
                            i += 1
                    else:
                        if len(line.strip()) > 0:
                            result_file.write(",".join(re.split(" +", line.strip())))
                            result_file.write("\n")


    @staticmethod
    def _parse_output_without_centroids(res, output_file):
        lines = res.decode().split("\n")

        i = 0
        with open(output_file, 'w') as result_file:
            for line in lines:
                matches = re.fullmatch(" {5}([0-9]+)", line)

                if matches is not None:
                    result_file.write("{},{}\n".format(i, matches.group(1)))
                    i += 1

    def _build_base_command(self, command):
        seed_setup_command = "'shuffle'" if self.seed is None else "{}".format(self.seed)
        return [MATLAB_EXE, "-nodisplay", "-nosplash", "-nodesktop",
                "-r \"{}rng({}); {} exit;\"   ".format(MatLab._matlab_redirect_temp_folder(TEMPFOLDER),
                                                     seed_setup_command, command)]

    def _build_command(self, command):
        """
        Build a matlab command with needed flags for a proper run
        :param command: The base command to execute
        """
        ret = self._build_base_command("[idx,C] = {}; disp(idx); disp('===C==='); disp(num2str(C));".format(command))

        if self.debug:
            print(ret)

        return ret

    def _build_command_without_centroids(self, command):
        """
        Build a matlab command with needed flags for a proper run. Do not print any centroid info.
        :param command: The base command to execute
        """
        ret = self._build_base_command("idx = {}; disp(idx);".format(command))

        if self.debug:
            print(ret)

        return ret

    def run_kmeans(self, nb_clusters, src_file, data_without_target, dataset_name, initial_clusters_file,
                   initial_clusters, run_number, run_info=None, nb_iterations=None):
        output_file, centroids_file = self._prepare_files(dataset_name, run_info, True)

        temp_file = self._dump_data_on_clean_csv(data_without_target)

        initial_clusters_matlab_string_matrix = []
        for aClusterFeatures in initial_clusters:
            initial_clusters_matlab_string_matrix.append(",".join(map(lambda x: str(x), aClusterFeatures.tolist())))
        initial_clusters_matlab_string_matrix = ";".join(initial_clusters_matlab_string_matrix)

        matlab_command_more = ''
        if nb_iterations is not None:
            matlab_command_more = ", 'MaxIter', {}".format(nb_iterations)

        matlab_command = "kmeans(csvread('{}'), {}, 'Start', [{}]{})".format(temp_file, str(nb_clusters),
                                                                             initial_clusters_matlab_string_matrix,
                                                                             matlab_command_more)

        command_parts = self._build_command(matlab_command)
        result = subprocess.run(command_parts, stdout=subprocess.PIPE)
        res = result.stdout

        MatLab._parse_output(res, output_file, centroids_file)
        unlink(temp_file)

        return output_file, {"centroids": centroids_file}

    def run_kmeans_plus_plus(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None,
                             nb_iterations=None):
        output_file, centroids_file = self._prepare_files(dataset_name, run_info, True)

        temp_file = self._dump_data_on_clean_csv(data_without_target)

        matlab_command_more = ''
        if nb_iterations is not None:
            matlab_command_more = ', MaxIter, {}'.format(nb_iterations)

        matlab_command = "kmeans(csvread('{}'), {}{})".format(temp_file, str(nb_clusters), matlab_command_more)

        command_parts = self._build_command(matlab_command)
        result = subprocess.run(command_parts, stdout=subprocess.PIPE)
        res = result.stdout

        MatLab._parse_output(res, output_file, centroids_file)
        unlink(temp_file)

        return output_file, {"centroids": centroids_file}

    def run_hierarchical(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None):
        output_file, = self._prepare_files(dataset_name, run_info, False)

        temp_file = ClusteringToolkit._dump_data_on_clean_csv(data_without_target)
        matlab_command = "cluster(linkage(csvread('{}'), 'ward'),'Maxclust',{})".format(temp_file, str(nb_clusters))
        command_parts = self._build_command_without_centroids(matlab_command)

        result = subprocess.run(command_parts, stdout=subprocess.PIPE)
        res = result.stdout
        MatLab._parse_output_without_centroids(res, output_file)
        unlink(temp_file)

        return output_file, {}

    def run_gaussian(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None):
        output_file, = self._prepare_files(dataset_name, run_info, False)
        temp_file = ClusteringToolkit._dump_data_on_clean_csv(data_without_target)

        matlab_command = "cluster(fitgmdist(csvread('{}'),{},'RegularizationValue',0.1), csvread('{}'))"\
            .format(temp_file, str(nb_clusters), temp_file)
        command_parts = self._build_command_without_centroids(matlab_command)
        result = subprocess.run(command_parts, stdout=subprocess.PIPE)
        res = result.stdout
        MatLab._parse_output_without_centroids(res, output_file)
        unlink(temp_file)

        return output_file, {}
