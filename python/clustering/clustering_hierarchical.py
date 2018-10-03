# Entry point for generating Gaussian Mixture Clusters
# Author: Vincenzo Musco (http://www.vmusco.com)


import clustering.hierarchical.run_base as run
from clustering.main_clustering import ClusterPipeline
from tools.clustering_constants import INCLUDED_ALGO
from tools.static import HIERARCHICAL_ALGO

RUN_INFO_BASE = HIERARCHICAL_ALGO
AVAIL_ALGOS = INCLUDED_ALGO[RUN_INFO_BASE]

class Hierarchical(ClusterPipeline):

    def __init__(self):
        super().__init__(RUN_INFO_BASE, AVAIL_ALGOS, run)



Hierarchical().runPipe()