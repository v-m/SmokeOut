# This file just defines some values needed for generating tables
# cf. Xin for exact implementation
# Author: Vincenzo Musco (http://www.vmusco.com)

import mlperf.clustering.spectralclustering.run_base as run
from mlperf.clustering.main_clustering import ClusterPipeline
from mlperf.tools.static import SPECTRAL_ALGO, INCLUDED_ALGO

RUN_INFO_BASE = SPECTRAL_ALGO
AVAIL_ALGOS = INCLUDED_ALGO[RUN_INFO_BASE]

class Spectralclustering(ClusterPipeline):

    def __init__(self):
        super().__init__(RUN_INFO_BASE, AVAIL_ALGOS, run)

Spectralclustering().runPipe()