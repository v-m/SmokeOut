"""Constants"""

__author__ = "Vincenzo Musco (http://www.vmusco.com)"

from mlperf import get_data

# TOOLKITS CONSTANTS
MATLAB_ALGO = "matlab"
OCTAVE_ALGO = "octave"
WEKA_ALGO = "weka"
WEKA_UNORM_ALGO = "weka_unorm"
MLPACK_ALGO = "mlpack"
R_ALGO = "R"
R_100ITER_ALGO = "R.100_iter"
TENSORFLOW_ALGO = "tensorflow"
SKLEARN_ALGO = "sklearn"
SKLEARN__FAST_ALGO = "sklearnfast"
SKLEARN_TOL0_ALGO = "sklearn.0_tol"
SHOGUN_ALGO = "shogun"
OPENCV_ALGO = "opencv"

# ALGORITHMS CONSTANTS
AFFINITY_PROP_ALGO = "apcluster"
DBSCAN_ALGO = "DBSCAN"
GAUSSIANMIX_ALGO = "gaussianm"
HIERARCHICAL_ALGO = "hierarchical"
KMEANS_ALGO = "fixed_clusters"
KMEANS_PLUSPLUS_ALGO = "kpp_run"
MEANSHIFT_ALGO = "meanshift"
SPECTRAL_ALGO = "spectralclustering"

JAVA_CLASSPATH = [
    get_data('java/main'),
    get_data('java/lib/weka-stable-3.8.0.jar'),
    get_data('java/lib/bounce-0.18.jar')
]

JAVA_CLASSPATH = ":".join(JAVA_CLASSPATH)
R_SCRIPT_BASE_DIR = get_data('R')