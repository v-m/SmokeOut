# Entry point for generating k-means++
# Author: Vincenzo Musco (http://www.vmusco.com)

import clustering.kmeans.run_base as run
from clustering.main_clustering import ClusterPipeline
from tools.clustering_constants import INCLUDED_ALGO
from tools.static import KMEANS_PLUSPLUS_ALGO, WEKA_UNORM_ALGO, SKLEARN_TOL0_ALGO

RUN_INFO_BASE = KMEANS_PLUSPLUS_ALGO
AVAIL_ALGOS = INCLUDED_ALGO[RUN_INFO_BASE]

class KMeansPlusPlus(ClusterPipeline):
    def __init__(self):
        super().__init__(RUN_INFO_BASE, AVAIL_ALGOS, run)

    def otherProcessRun(self, ALGO, srcFile, clustersNumber, dataLessTarget, datasetName, RUN_INFO):
        # SCIKIT
        if ALGO == SKLEARN_TOL0_ALGO:
            run.sklearnProcess(clustersNumber, dataLessTarget, datasetName, RUN_INFO, zeroTolerance=True)

            # sc = pyspark.SparkContext("local", "first app")
            # sparkKmeanModel = pyspark.ml.clustering.KMeans().setK(0).setSeed(1)
            # sparkKmeanModelRun = sparkKmeanModel.fit(dataLessTarget)

            # TENSORFLOW (Expecting a class/label ?!?)
            # tensorflowKmeansModel = tf.contrib.factorization.KMeansClustering(num_clusters=clustersNumber, use_mini_batch=False, random_seed=1, initial_clusters=tf.contrib.factorization.KMeansClustering.KMEANS_PLUS_PLUS_INIT)
            # tensorflowKmeansModel.train(dataLessTarget)

            # print(computeMD5ResultIdentity(sklearnKmeanModel.labels_)
        elif ALGO == WEKA_UNORM_ALGO:
            run.wekaUnormProcess(srcFile, datasetName, RUN_INFO)

KMeansPlusPlus().runPipe()
