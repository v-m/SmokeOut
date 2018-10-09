# Entry point for generating Gaussian Mixture Clusters
# Author: Vincenzo Musco (http://www.vmusco.com)


import mlperf.clustering.hierarchical.run_base as run
from mlperf.clustering.main_clustering import ClusterPipeline
from mlperf.tools.static import HIERARCHICAL_ALGO, INCLUDED_ALGO

RUN_INFO_BASE = HIERARCHICAL_ALGO
AVAIL_ALGOS = INCLUDED_ALGO[RUN_INFO_BASE]

class Hierarchical(ClusterPipeline):

    def __init__(self):
        super().__init__(RUN_INFO_BASE, AVAIL_ALGOS, run)



Hierarchical().runPipe()