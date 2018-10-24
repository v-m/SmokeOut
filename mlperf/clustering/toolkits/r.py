"""R clustering"""

__author__ = "Vincenzo Musco (http://www.vmusco.com)"

from os import path
from subprocess import Popen, PIPE

from mlperf.clustering import clusteringtoolkit
from mlperf.tools.config import R_BIN
from mlperf.tools.static import R_ALGO
import pandas


class R(clusteringtoolkit.ClusteringToolkit):
    def toolkit_name(self):
        return R_ALGO

    def check_toolkit_requirements(self):
        if not path.exists(R_BIN):
            raise FileNotFoundError("Unable to locate R binary")

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

    def run_kmeans(self, nb_clusters, src_file, data_without_target, dataset_name, initial_clusters_file,
                   initial_clusters, run_number, run_info=None, nb_iterations=None):
        # if initialClustersCsvFile is None and hundredIters:
        #     print("R kcca function don't have the iteration parameter !")
        #     return

        output_file, centroids_file = self._prepare_files(dataset_name, run_info, True)

        r_script = self._build_kmeans_script(src_file, output_file, centroids_file, initial_clusters_file,
                                             nb_iterations,
                                             self.seed)
        p = Popen([R_BIN, '--vanilla'], stdout=PIPE, stdin=PIPE, stderr=PIPE)
        p.communicate(input=r_script)

        dta = pandas.read_csv(centroids_file)
        dta.drop(dta.columns[[0]], axis=1).to_csv(centroids_file, index=False, header=False)

        return output_file, {"centroids": centroids_file}

    def run_kmeans_plus_plus(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None,
                             nb_iterations=None):
        output_file, centroids_file = self._prepare_files(dataset_name, run_info, True)

        r_script = self._build_kmeans_script(src_file, output_file, centroids_file, None, nb_iterations, self.seed)
        p = Popen([R_BIN, '--vanilla'], stdout=PIPE, stdin=PIPE, stderr=PIPE)
        p.communicate(input=r_script)

        dta = pandas.read_csv(centroids_file)
        dta.drop(dta.columns[[0]], axis=1).to_csv(centroids_file, index=False, header=False)

        return output_file, {"centroids": centroids_file}

    def run_hierarchical(self, nb_clusters, src_file, data_without_target, dataset_name, run_number, run_info=None):
        output_file, = self._prepare_files(dataset_name, run_info, False)

        r_script = self._build_hierarchical(src_file, output_file, self.seed)
        p = Popen([R_BIN, '--vanilla'], stdout=PIPE, stdin=PIPE, stderr=PIPE)
        p.communicate(input=r_script)

        return output_file, {}
