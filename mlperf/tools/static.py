"""Constants"""

__author__ = "Vincenzo Musco (http://www.vmusco.com)"

from mlperf import get_data

# TOOLKITS CONSTANTS
MATLAB_TOOLKIT = "matlab"
OCTAVE_TOOLKIT = "octave"
WEKA_TOOLKIT = "weka"
WEKA_UNORM_TOOLKIT = "weka_unorm"
MLPACK_TOOLKIT = "mlpack"
R_TOOLKIT = "R"
TENSORFLOW_TOOLKIT = "tensorflow"
SKLEARN_TOOLKIT = "sklearn"
SKLEARN__FAST_TOOLKIT = "sklearnfast"
SKLEARN_TOL0_TOOLKIT = "sklearn.0_tol"
SHOGUN_TOOLKIT = "shogun"
OPENCV_TOOLKIT = "opencv"

# ALGORITHMS CONSTANTS
# Not used anymore. 
# AFFINITY_PROP_ALGO = "apcluster"
# DBSCAN_ALGO = "DBSCAN"
# GAUSSIANMIX_ALGO = "gaussianm"
# HIERARCHICAL_ALGO = "hierarchical"
# KMEANS_ALGO = "fixed_clusters"
# KMEANS_PLUSPLUS_ALGO = "kpp_run"
# MEANSHIFT_ALGO = "meanshift"
# SPECTRAL_ALGO = "spectralclustering"

JAVA_CLASSPATH = ":".join([
    get_data('java/main'),
    get_data('java/lib/weka-stable-3.8.0.jar'),
    get_data('java/lib/bounce-0.18.jar')
])
