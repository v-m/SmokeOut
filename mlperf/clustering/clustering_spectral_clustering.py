# This file just defines some values needed for generating tables
# cf. Xin for exact implementation
# Author: Vincenzo Musco (http://www.vmusco.com)

from mlperf.tools.static import SPECTRAL_ALGO, INCLUDED_ALGO

RUN_INFO_BASE = SPECTRAL_ALGO
AVAIL_ALGOS = INCLUDED_ALGO[RUN_INFO_BASE]
