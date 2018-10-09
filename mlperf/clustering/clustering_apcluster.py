# This file just defines some values needed for generating tables
# cf. Xin for exact implementation
# Author: Vincenzo Musco (http://www.vmusco.com)

import mlperf.clustering.affinitypropagation.run_base as run
from mlperf.clustering.main_clustering import ClusterPipeline
from mlperf.tools.static import AFFINITY_PROP_ALGO, INCLUDED_ALGO

RUN_INFO_BASE = AFFINITY_PROP_ALGO
AVAIL_ALGOS = INCLUDED_ALGO[RUN_INFO_BASE]

class AffinityPropagation(ClusterPipeline):

    def __init__(self):
        super().__init__(RUN_INFO_BASE, AVAIL_ALGOS, run)

AffinityPropagation().runPipe()