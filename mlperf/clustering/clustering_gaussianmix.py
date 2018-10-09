# Entry point for generating Gaussian Mixture Clusters
# Author: Vincenzo Musco (http://www.vmusco.com)

import mlperf.clustering.gaussianmixture.run_base as run
from mlperf.clustering.main_clustering import ClusterPipeline
from mlperf.tools.static import GAUSSIANMIX_ALGO, SKLEARN_TOL0_ALGO, INCLUDED_ALGO

RUN_INFO_BASE = GAUSSIANMIX_ALGO
AVAIL_ALGOS = INCLUDED_ALGO[RUN_INFO_BASE]

class GaussianMixture(ClusterPipeline):
    def __init__(self):
        super().__init__(RUN_INFO_BASE, AVAIL_ALGOS, run)


    def otherProcessRun(self, ALGO, srcFile, clustersNumber, dataLessTarget, datasetName, RUN_INFO):
        # SCIKIT 0tol
        if ALGO == SKLEARN_TOL0_ALGO:
            run.sklearnProcess(clustersNumber, dataLessTarget, datasetName, RUN_INFO, zeroTolerance=True)


GaussianMixture().runPipe()