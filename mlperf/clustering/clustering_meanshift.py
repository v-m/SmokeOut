# This file just defines some values needed for generating tables
# cf. Xin for exact implementation
# Author: Vincenzo Musco (http://www.vmusco.com)

from mlperf.tools.static import INCLUDED_ALGO, MEANSHIFT_ALGO

RUN_INFO_BASE = MEANSHIFT_ALGO
AVAIL_ALGOS = INCLUDED_ALGO[RUN_INFO_BASE]
