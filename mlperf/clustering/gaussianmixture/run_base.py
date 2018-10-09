#
# Author: Vincenzo Musco (http://www.vmusco.com)

import csv
import os
import re
import subprocess

from mlperf.clustering.tools import dumpDataOnCleanCsv
from mlperf.tools.config import MATLAB_EXE, TEMPFOLDER, JAVA_EXE
from mlperf.tools.static import datasetOutFile, MATLAB_ALGO, matlabRedirectTempFolder, WEKA_ALGO, JAVA_CLASSPATH, \
    SKLEARN_TOL0_ALGO, SKLEARN_ALGO, TENSORFLOW_ALGO, centroidFor

def matlabProcess(clustersNumber, dataLessTarget, datasetName, runinfo = None):
    outputFile = datasetOutFile(datasetName, MATLAB_ALGO, runinfo=runinfo)

    if os.path.exists(outputFile):
        print("matlab skipped")
        return

    tempFile = dumpDataOnCleanCsv(dataLessTarget)

    matlabCmd = "cluster(fitgmdist(csvread('{}'),{},'RegularizationValue',0.1), csvread('{}'))".format(tempFile, str(clustersNumber), tempFile)
    command_parts = [MATLAB_EXE, "-nodisplay", "-nosplash", "-nodesktop", "-r \"rng('shuffle'); {}idx = {}; disp(idx); exit;\"   ".format(matlabRedirectTempFolder(TEMPFOLDER), matlabCmd)]
    print(" ".join(command_parts))
    result = subprocess.run(command_parts, stdout=subprocess.PIPE)
    res = result.stdout

    i = 0
    resultFile = open(outputFile, 'w')
    for line in res.decode().split("\n"):
        matches = re.fullmatch("     ?([0-9]+)", line)

        if matches is not None:
            resultFile.write("{},{}\n".format(i, matches.group(1)))
            i += 1

    resultFile.close()

    os.unlink(tempFile)




def wekaProcess(inFile, datasetName, runinfo = None):
    outputFile = datasetOutFile(datasetName, WEKA_ALGO, runinfo = runinfo)

    if os.path.exists(outputFile):
        print("weka skipped")
        return

    command_parts = [JAVA_EXE, "-Xmx100g", "-classpath", JAVA_CLASSPATH, "EMWekaRun", inFile, outputFile]
    print(" ".join(command_parts))

    subprocess.call(command_parts)



def sklearnProcess(clustersNumber, dataLessTarget, datasetName, runinfo = None, zeroTolerance = False):
    # Local import
    import sklearn.mixture

    selectedAlgo = SKLEARN_TOL0_ALGO if zeroTolerance else SKLEARN_ALGO
    outputFile = datasetOutFile(datasetName, selectedAlgo, runinfo=runinfo)

    if os.path.exists(outputFile):
        print("sklearn {}skipped".format("Zero Tol. " if zeroTolerance else ""))
        return

    builtModel = None

    if zeroTolerance:
        builtModel = sklearn.mixture.GaussianMixture(n_components=clustersNumber, tol=0)
    else:
        builtModel = sklearn.mixture.GaussianMixture(n_components=clustersNumber)

    builtModel.fit(dataLessTarget)

    predictedLabels = builtModel.predict(dataLessTarget)

    with open(outputFile, 'w') as csvfile:
        filewriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)

        for index, row in dataLessTarget.iterrows():
            filewriter.writerow([index, predictedLabels[index]])


# def shogunProcess(clustersNumber, dataLessTarget, datasetName, runinfo = None):
#     outputFile = datasetOutFile(datasetName, SHOGUN_ALGO, runinfo=runinfo)
#
#     if os.path.exists(outputFile):
#         print("shogun skipped")
#         return
#
#     train_features = shogun.RealFeatures(dataLessTarget.values.astype("float64").transpose())
#     # distance metric over feature matrix - Euclidean distance
#     # distance = shogun.EuclideanDistance(train_features, train_features)
#
#     gmm = shogun.GMM(clustersNumber)
#     gmm.set_features(train_features)
#     gmm.train_em()
#
#     print(gmm)




def tensorflowProcess(clustersNumber, dataLessTarget, datasetName, runinfo = None, initialClusters = None):
    # Local import
    import tensorflow as tf
    from tensorflow.python.framework import constant_op
    import numpy as np
    '''
    https://www.tensorflow.org/api_docs/python/tf/contrib/factorization/KMeansClustering
    '''
    outputFile = datasetOutFile(datasetName, TENSORFLOW_ALGO, runinfo=runinfo)
    clustersOutputFile = datasetOutFile(datasetName, centroidFor(TENSORFLOW_ALGO), runinfo=runinfo)

    if os.path.exists(outputFile) and os.path.exists(clustersOutputFile):
        print("tensorflow skipped")
        return

    points = dataLessTarget.values

    def get_input_fn():
        def input_fn():
            return constant_op.constant(points.astype(np.float32)), None

        return input_fn

    if initialClusters is None:
        gmm = tf.contrib.factorization.GMM(num_clusters=clustersNumber)
    else:
        gmm = tf.contrib.factorization.GMM(num_clusters=clustersNumber, initial_clusters=initialClusters)

    gmm.fit(input_fn=get_input_fn(), steps=1)


    with open(outputFile, 'w') as csvfile:
        filewriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)

        cluster_indices = list(gmm.predict_assignments())
        for index, point in enumerate(points):
            cluster_index = cluster_indices[index]
            filewriter.writerow([index, cluster_index])

    # Clusters saving
    with open(clustersOutputFile, 'w') as clusterFile:
        filewriter = csv.writer(clusterFile, quoting=csv.QUOTE_MINIMAL)

        for row in gmm.cluster_centers():
            filewriter.writerow(row.tolist())

    # with open(clustersOutputFile, 'w') as clusterFile:
    #     kmeans.cluster_centers().tofile(clusterFile)