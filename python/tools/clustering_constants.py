from tools.static import *

INCLUDED_ALGO = {
    SPECTRAL_ALGO: [SKLEARN_ALGO, SKLEARN__FAST_ALGO, R_ALGO],
    MEANSHIFT_ALGO: [SKLEARN_ALGO],
    KMEANS_PLUSPLUS_ALGO: [SKLEARN_ALGO,
               SKLEARN_TOL0_ALGO,
               R_ALGO,
               R_100ITER_ALGO,
               MLPACK_ALGO,
               MATLAB_ALGO,
               WEKA_ALGO,
               WEKA_UNORM_ALGO,
               SHOGUN_ALGO,
               OPENCV_ALGO,
               TENSORFLOW_ALGO],
    KMEANS_ALGO: [SKLEARN_ALGO,
                SKLEARN_TOL0_ALGO,
                R_ALGO,
                MLPACK_ALGO,
                MATLAB_ALGO,
                SHOGUN_ALGO,
                R_100ITER_ALGO,
                TENSORFLOW_ALGO],
    HIERARCHICAL_ALGO: [
                SKLEARN_ALGO
                # , WEKA_ALGO
                , R_ALGO
                , MATLAB_ALGO
                # , SHOGUN_ALGO  --> Shogun makes python crash (!?)
            ],
    GAUSSIANMIX_ALGO: [SKLEARN_ALGO, SKLEARN_TOL0_ALGO, MATLAB_ALGO, WEKA_ALGO, TENSORFLOW_ALGO],
    DBSCAN_ALGO: [SKLEARN_ALGO, R_ALGO, MLPACK_ALGO],
    AFFINITY_PROP_ALGO: [SKLEARN_ALGO, R_ALGO]
 }