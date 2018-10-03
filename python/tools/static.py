#
# Author: Vincenzo Musco (http://www.vmusco.com)
import os
import sklearn.metrics

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



JAVA_CLASSPATH = ":".join([
                    os.path.abspath("{}/../../java/out/production/classes".format(os.path.dirname(__file__))),
                    os.path.abspath("{}/../../java/deps/weka-stable-3.8.0.jar".format(os.path.dirname(__file__))),
                    os.path.abspath("{}/../../java/deps/bounce-0.18.jar".format(os.path.dirname(__file__)))
                ])


SCORING_METRICS = [sklearn.metrics.adjusted_rand_score,

                   sklearn.metrics.adjusted_mutual_info_score,
                   sklearn.metrics.normalized_mutual_info_score,

                   sklearn.metrics.homogeneity_score,
                   sklearn.metrics.completeness_score,
                   sklearn.metrics.v_measure_score,

                   sklearn.metrics.fowlkes_mallows_score]

def exploreDatasets(root):
    for datasetName in os.listdir(root):
        if os.path.isdir(os.path.join(root, datasetName)):
            yield datasetName


def datasetOutFile(datasetName, algoName, ext = "csv", runinfo = None):
    return "{}.{}{}.{}".format(datasetName, algoName, ".{}".format(runinfo) if runinfo is not None else "", ext)


def centroidFor(algoName):
    return "{}.centroids".format(algoName)

def runForNr(runBase, runId):
    return "{}{}".format(runBase, runId)

def matlabRedirectTempFolder(tempdir):
    return "tempdir='{}';".format(tempdir)