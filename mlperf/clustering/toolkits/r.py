"""R clustering"""
__author__ = "Vincenzo Musco (http://www.vmusco.com)"

import re
import subprocess
from os import path

from mlperf.clustering import clusteringtoolkit
from mlperf.tools.config import R_BIN
from mlperf.tools.static import R_TOOLKIT
import pandas


class R(clusteringtoolkit.ClusteringToolkit):
    def toolkit_name(self):
        return R_TOOLKIT

    def check_toolkit_requirements(self):
        if not path.exists(R_BIN):
            raise FileNotFoundError("Unable to locate R binary")

    @staticmethod
    def install_package(package_name):
        script_parts = '''install.packages("{}");'''.format(package_name)
        p = subprocess.Popen([R_BIN, '--vanilla'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        p.communicate(input=script_parts.encode())
        return p.returncode

    @staticmethod
    def uninstall_package(package_name):
        script_parts = '''remove.packages("{}");'''.format(package_name)
        p = subprocess.Popen([R_BIN, '--vanilla'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        p.communicate(input=script_parts.encode())
        return p.returncode

    def _build_kmeans_script(self, src_file, dst_clusters, dst_centroids, init_clusters=None, max_iter=None, seed=None):
        script_parts = []

        if seed is not None:
            script_parts.append('''set.seed({})'''.format(seed))

        script_parts.append('''source_file=gzfile("{}");'''.format(src_file))
        script_parts.append('''dat=read.csv(source_file, header=T, sep='\t')''')

        # Let's remove the target feature
        script_parts.append('''clustersNumber = nrow(unique(dat["target"]))''')
        script_parts.append('''datWithoutTarget = subset( dat, select = -target )''')

        # http://stat.ethz.ch/R-manual/R-devel/library/stats/html/kmeans.html
        iter_string = ""
        if max_iter is not None:
            iter_string = ", iter.max = {}".format(max_iter)

        if init_clusters is not None:
            script_parts.append('''init_clusters = read.csv('{}', header = FALSE)'''.format(init_clusters))
            script_parts.append('''clusteringResult = kmeans(datWithoutTarget, init_clusters{})'''.format(iter_string))
        else:
            script_parts.append('''clusteringResult = kmeans(datWithoutTarget, clustersNumber{})'''.format(iter_string))

        script_parts.append('''write.csv(clusteringResult["cluster"], file='{}')'''.format(dst_clusters))
        script_parts.append('''write.csv(clusteringResult["centers"], file='{}')'''.format(dst_centroids))

        # clusters = clusteringResult$cluster
        # print(clusteringResult)
        # print(clusters(clusteringResult))

        # [1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
        # [6] "betweenss"    "size"         "iter"         "ifault"

        ret = "\n".join(script_parts).encode()

        if self.debug:
            print(ret)

        return ret

    def _build_hierarchical(self, src_file, dst_clusters, seed=None):
        script_parts = []

        if seed is not None:
            script_parts.append('''set.seed({})'''.format(seed))

        script_parts.append('''library(cluster);'''.format(seed))

        script_parts.append('''source_file=gzfile("{}");'''.format(src_file))
        script_parts.append('''dat=read.csv(source_file, header=T, sep='\t');''')

        # Let's remove the target feature
        script_parts.append('''clustersNumber = nrow(unique(dat["target"]));''')
        script_parts.append('''datWithoutTarget = subset( dat, select = -target );''')

        script_parts.append('''d <- dist(datWithoutTarget, method = "euclidean");''')
        script_parts.append('''hc1 <- hclust(d, method = "ward.D2" );''')
        script_parts.append('''sub_grp <- cutree(hc1, k = clustersNumber);''')

        script_parts.append('''write.csv(clusteringResult["cluster"], file='{}');'''.format(dst_clusters))

        ret = "\n".join(script_parts).encode()

        if self.debug:
            print(ret)

        return ret

    @staticmethod
    def _package_version(package_name):
        script_parts = ['''packageVersion("{}");'''.format(package_name)]

        p = subprocess.Popen([R_BIN, '--vanilla'], stdout=subprocess.PIPE, stdin=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        out, err = p.communicate(input="\n".join(script_parts).encode())
        groups = re.fullmatch(".*?\[1\] ‘([^’]+)’.*?", out.decode(), flags=re.DOTALL)
        return groups.group(1)

    def run_kmeans(self, nb_clusters, src_file, data_without_target, dataset_name, initial_clusters_file,
                   initial_clusters, run_number, run_info=None, nb_iterations=None):
        output_file, centroids_file = self._prepare_files(dataset_name, run_info, True)

        r_script = self._build_kmeans_script(src_file, output_file, centroids_file, initial_clusters_file,
                                             nb_iterations, self.seed)
        p = subprocess.Popen([R_BIN, '--vanilla'], stdout=subprocess.PIPE, stdin=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        p.communicate(input=r_script)

        dta = pandas.read_csv(centroids_file)
        dta.drop(dta.columns[[0]], axis=1).to_csv(centroids_file, index=False, header=False)

        return output_file, {"centroids": centroids_file}

    def run_kmeans_random(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None,
                          nb_iterations=None):
        output_file, centroids_file = self._prepare_files(dataset_name, run_info, True)

        r_script = self._build_kmeans_script(src_file, output_file, centroids_file, None, nb_iterations, self.seed)
        p = subprocess.Popen([R_BIN, '--vanilla'], stdout=subprocess.PIPE, stdin=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        p.communicate(input=r_script)

        dta = pandas.read_csv(centroids_file)
        dta.drop(dta.columns[[0]], axis=1).to_csv(centroids_file, index=False, header=False)

        return output_file, {"centroids": centroids_file}

    def run_hierarchical(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None):
        output_file, = self._prepare_files(dataset_name, run_info, False)

        r_script = self._build_hierarchical(src_file, output_file, self.seed)
        p = subprocess.Popen([R_BIN, '--vanilla'], stdout=subprocess.PIPE, stdin=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        p.communicate(input=r_script)

        return output_file, {}


class RClusterR(R):
    def _build_kmeanspp_script_clusterer(self, src_file, dst_clusters, dst_centroids, max_iter=None, seed=None):
        """
        Run kpp using the ClusterR package and KMeans_rcpp function.
        """
        script_parts = ['''library("ClusterR")''']
        # KMeans_rcpp
        if seed is not None:
            script_parts.append('''set.seed({})'''.format(seed))

        script_parts.append('''source_file=gzfile("{}");'''.format(src_file))
        script_parts.append('''dat=read.csv(source_file, header=T, sep='\t')''')

        # Let's remove the target feature
        script_parts.append('''clustersNumber = nrow(unique(dat["target"]))''')
        script_parts.append('''datWithoutTarget = subset( dat, select = -target )''')

        # https://www.rdocumentation.org/packages/flexclust/versions/1.3-5/topics/kcca
        script_part_more = ''

        if max_iter is not None:
            script_part_more = '{},max_iters = {}'.format(script_part_more, max_iter)

        script_parts.append('''clusteringResult = KMeans_rcpp(datWithoutTarget, clusters = clustersNumber, '''
                            '''initializer='kmeans++', verbose = F{})'''.format(script_part_more))

        script_parts.append('''write.csv(clusteringResult$clusters, file='{}')'''.format(dst_clusters))
        script_parts.append('''write.csv(clusteringResult$centroids, file='{}')'''.format(dst_centroids))

        ret = "\n".join(script_parts).encode()

        if self.debug:
            print(ret)

        return ret

    def package_version(self):
        return R._package_version("ClusterR")

    def run_kmeans_plus_plus(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None,
                             nb_iterations=None):
        output_file, centroids_file = self._prepare_files(dataset_name, run_info, True)

        r_script = self._build_kmeanspp_script_clusterer(src_file, output_file, centroids_file, nb_iterations, self.seed)
        p = subprocess.Popen([R_BIN, '--vanilla'], stdout=subprocess.PIPE, stdin=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        p.communicate(input=r_script)

        dta = pandas.read_csv(centroids_file)
        dta.drop(dta.columns[[0]], axis=1).to_csv(centroids_file, index=False, header=False)

        return output_file, {"centroids": centroids_file}


class RFlexclust(R):
    def _build_kmeanspp_script_flexclust(self, src_file, dst_clusters, dst_centroids, seed=None):
        """
        Run kpp using the flexclust package and kcca function.
        """
        script_parts = ['''library("flexclust")''']

        if seed is not None:
            script_parts.append('''set.seed({})'''.format(seed))

        script_parts.append('''source_file=gzfile("{}");'''.format(src_file))
        script_parts.append('''dat=read.csv(source_file, header=T, sep='\t')''')

        # Let's remove the target feature
        script_parts.append('''clustersNumber = nrow(unique(dat["target"]))''')
        script_parts.append('''datWithoutTarget = subset( dat, select = -target )''')

        # https://www.rdocumentation.org/packages/flexclust/versions/1.3-5/topics/kcca
        script_parts.append('clusteringResult = kcca(datWithoutTarget, clustersNumber, family=kccaFamily("kmeans"),'
                            'control=list(initcent="kmeanspp"), simple=F)')

        script_parts.append('''write.csv(clusters(clusteringResult), file='{}')'''.format(dst_clusters))
        script_parts.append('''write.csv(parameters(clusteringResult), file='{}')'''.format(dst_centroids))

        ret = "\n".join(script_parts).encode()

        if self.debug:
            print(ret)

        return ret

    def package_version(self):
        return R._package_version("flexclust")

    def run_kmeans_plus_plus(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None,
                             nb_iterations=None):
        output_file, centroids_file = self._prepare_files(dataset_name, run_info, True)

        r_script = self._build_kmeanspp_script_flexclust(src_file, output_file, centroids_file, self.seed)
        p = subprocess.Popen([R_BIN, '--vanilla'], stdout=subprocess.PIPE, stdin=subprocess.PIPE,
                             stderr=subprocess.PIPE)

        if self.debug:
            print(r_script)

        p.communicate(input=r_script)

        dta = pandas.read_csv(centroids_file)
        dta.drop(dta.columns[[0]], axis=1).to_csv(centroids_file, index=False, header=False)

        return output_file, {"centroids": centroids_file}
