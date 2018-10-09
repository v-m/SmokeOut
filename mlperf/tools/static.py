#
# Author: Vincenzo Musco (http://www.vmusco.com)
import os
from glob import glob

# TOOLKITS CONSTANTS
MATLAB_ALGO = "matlab"
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

# Dynamically building the JAVA_CLASSPATH for Weka
JAVA_CLASSPATH = list(map(lambda x: os.path.dirname(x), glob("**/EMWekaRun.class", recursive=True)))
JAVA_CLASSPATH.extend(glob("**/weka-stable-3.8.0.jar", recursive=True))
JAVA_CLASSPATH.extend(glob("**/bounce-0.18.jar", recursive=True))

JAVA_CLASSPATH = ":".join(JAVA_CLASSPATH)

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


def exploreDatasets(root):
    for datasetName in os.listdir(root):
        if os.path.isdir(os.path.join(root, datasetName)):
            yield datasetName


def datasetOutFile(datasetName, algoName, ext="csv", runinfo=None):
    return "{}.{}{}.{}".format(datasetName, algoName, ".{}".format(runinfo) if runinfo is not None else "", ext)


def centroidFor(algoName):
    return "{}.centroids".format(algoName)


def runForNr(runBase, runId):
    return "{}{}".format(runBase, runId)


def matlabRedirectTempFolder(tempdir):
    return "tempdir='{}';".format(tempdir)
