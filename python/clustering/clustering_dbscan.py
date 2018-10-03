# This file just defines some values needed for generating tables
# cf. Xin for exact implementation
# Author: Vincenzo Musco (http://www.vmusco.com)
from tools.clustering_constants import INCLUDED_ALGO
from tools.static import DBSCAN_ALGO

RUN_INFO_BASE = DBSCAN_ALGO
AVAIL_ALGOS = INCLUDED_ALGO[RUN_INFO_BASE]
