"""Matlab clustering"""

__author__ = "Vincenzo Musco (http://www.vmusco.com)"

from os import path
import subprocess

from mlperf.clustering import clusteringtoolkit
from mlperf.tools.config import MATLAB_EXE, TEMPFOLDER, OCTAVE_EXE
from mlperf.tools.static import MATLAB_ALGO, OCTAVE_ALGO


class MatLab(clusteringtoolkit.ClusteringToolkit):
    def toolkit_name(self):
        return MATLAB_ALGO

    def check_toolkit_requirements(self):
        # Checking requirements on import
        if not path.exists(MATLAB_EXE):
            raise FileNotFoundError("Unable to locate matlab installation directory")

    @staticmethod
    def _matlab_redirect_temp_folder(temp_dir):
        return "tempdir='{}';".format(temp_dir)

    @staticmethod
    def _parse_external_output(produced_clustering_file, output_file):
        i = 0

        with open(output_file, 'w') as result_file:
            with open(produced_clustering_file, 'r') as src_file:
                source_content = src_file.read()
                for line in source_content.split("\n"):
                    if len(line.strip()) > 0:
                        result_file.write("{},{}\n".format(i, line))
                        i += 1

    def _build_base_command(self, command):
        seed_setup_command = "'shuffle'" if self.seed is None else "{}".format(self.seed)
        return [MATLAB_EXE, "-nodisplay", "-nosplash", "-nodesktop",
                "-r \"{}rng({}); {} exit;\"   ".format(MatLab._matlab_redirect_temp_folder(TEMPFOLDER),
                                                       seed_setup_command, command)]

    def _build_command(self, command, result_to, centroids_file):
        """
        Build a matlab command with needed flags for a proper run
        :param command: The base command to execute
        """
        ret = self._build_base_command("[idx,C] = {}; csvwrite('{}', idx); csvwrite('{}', C);".format(command,
                                                                                                      result_to,
                                                                                                      centroids_file))
        if self.debug:
            print(ret)

        return ret

    def _build_command_without_centroids(self, command, result_to):
        """
        Build a matlab command with needed flags for a proper run. Do not print any centroid info.
        :param command: The base command to execute
        """
        ret = self._build_base_command("idx = {}; csvwrite('{}', idx);".format(command, result_to))

        if self.debug:
            print(ret)

        return ret

    def run_kmeans(self, nb_clusters, src_file, data_without_target, dataset_name, initial_clusters_file,
                   initial_clusters, run_number, run_info=None, nb_iterations=None):
        output_file, centroids_file = self._prepare_files(dataset_name, run_info, True)

        temp_file = self._dump_data_on_clean_csv(data_without_target)
        temp_file_out = self.create_temporary_file()

        initial_clusters_string_matrix = []
        for a_cluster_features in initial_clusters:
            initial_clusters_string_matrix.append(",".join(map(lambda x: str(x), a_cluster_features.tolist())))
        initial_clusters_string_matrix = ";".join(initial_clusters_string_matrix)

        built_command_more = ''
        if nb_iterations is not None:
            built_command_more = ", 'MaxIter', {}".format(nb_iterations)

        built_command = "kmeans(csvread('{}'), {}, 'Start', [{}]{})".format(temp_file, str(nb_clusters),
                                                                            initial_clusters_string_matrix,
                                                                            built_command_more)

        command_parts = self._build_command(built_command, temp_file_out, centroids_file)
        subprocess.run(command_parts, stdout=subprocess.PIPE)

        MatLab._parse_external_output(temp_file_out, output_file)
        self.clean_temporary_files()

        return output_file, {"centroids": centroids_file}

    def run_kmeans_plus_plus(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None,
                             nb_iterations=None):
        output_file, centroids_file = self._prepare_files(dataset_name, run_info, True)

        temp_file = self._dump_data_on_clean_csv(data_without_target)
        temp_file_out = self.create_temporary_file()

        built_command_more = ''
        if nb_iterations is not None:
            built_command_more = ', MaxIter, {}'.format(nb_iterations)

        built_command = "kmeans(csvread('{}'), {}{})".format(temp_file, str(nb_clusters), built_command_more)

        command_parts = self._build_command(built_command, temp_file_out, centroids_file)
        subprocess.run(command_parts, stdout=subprocess.PIPE)

        MatLab._parse_external_output(temp_file_out, output_file)
        self.clean_temporary_files()

        return output_file, {"centroids": centroids_file}

    def run_hierarchical(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None):
        output_file, = self._prepare_files(dataset_name, run_info, False)

        temp_file = self._dump_data_on_clean_csv(data_without_target)
        temp_file_out = self.create_temporary_file()

        built_command = "cluster(linkage(csvread('{}'), 'ward'),'Maxclust',{})".format(temp_file, str(nb_clusters))
        command_parts = self._build_command_without_centroids(built_command, temp_file_out)

        subprocess.run(command_parts, stdout=subprocess.PIPE)
        MatLab._parse_external_output(temp_file_out, output_file)
        self.clean_temporary_files()

        return output_file, {}

    def run_gaussian(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None):
        output_file, = self._prepare_files(dataset_name, run_info, False)
        temp_file = self._dump_data_on_clean_csv(data_without_target)
        temp_file_out = self.create_temporary_file()

        built_command = "cluster(fitgmdist(csvread('{}'),{},'RegularizationValue',0.1), csvread('{}'))" \
            .format(temp_file, str(nb_clusters), temp_file)
        command_parts = self._build_command_without_centroids(built_command, temp_file_out)
        subprocess.run(command_parts, stdout=subprocess.PIPE)

        MatLab._parse_external_output(temp_file_out, output_file)
        self.clean_temporary_files()

        return output_file, {}


class Octave(MatLab):
    def toolkit_name(self):
        return OCTAVE_ALGO

    def check_toolkit_requirements(self):
        if not path.exists(OCTAVE_EXE):
            raise FileNotFoundError("Unable to locate octave installation directory")

    def _build_base_command(self, command):
        seed_setup_command = "" if self.seed is None else "rand(\"seed\", {});".format(self.seed)
        return [OCTAVE_EXE,
                "--eval", "pkg load statistics; {}{}{}".format(MatLab._matlab_redirect_temp_folder(TEMPFOLDER),
                                                               seed_setup_command, command)]

    def _build_command(self, command, result_to, centroids_file):
        """
        Build a matlab command with needed flags for a proper run
        :param command: The base command to execute
        """
        ret = self._build_base_command("[idx,C] = {}; csvwrite('{}', idx); csvwrite('{}', C);".format(command,
                                                                                                      result_to,
                                                                                                      centroids_file))

        if self.debug:
            print(ret)

        return ret

    def _build_command_without_centroids(self, command, result_to):
        """
        Build a matlab command with needed flags for a proper run. Do not print any centroid info.
        :param command: The base command to execute
        """
        ret = self._build_base_command("idx = {}; csvwrite('{}', idx);".format(command, result_to))

        if self.debug:
            print(ret)

        return ret

    @NotImplementedError
    def run_hierarchical(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None):
        pass
