#
# Author: Vincenzo Musco (http://www.vmusco.com)

import csv
import os
import subprocess
import time
import re
from mlperf.clustering.tools import dumpDataOnCleanCsv

# https://www.mathworks.com/help/stats/kmeans.html
from mlperf.tools.config import MATLAB_EXE, JAVA_EXE, TEMPFOLDER, MLPACK_BIN, R_BIN
from mlperf.tools.static import datasetOutFile, centroidFor, MATLAB_ALGO, JAVA_CLASSPATH, WEKA_ALGO, WEKA_UNORM_ALGO, \
    MLPACK_ALGO, SKLEARN_TOL0_ALGO, SHOGUN_ALGO, TENSORFLOW_ALGO, R_100ITER_ALGO, R_ALGO, OPENCV_ALGO, SKLEARN_ALGO, \
    R_SCRIPT_BASE_DIR, matlabRedirectTempFolder


def matlabProcess(clustersNumber, dataLessTarget, datasetName, runinfo = None, initialClusters = None):
    outputFile = datasetOutFile(datasetName, MATLAB_ALGO, runinfo=runinfo)
    clustersOutputFile = datasetOutFile(datasetName, centroidFor(MATLAB_ALGO), runinfo=runinfo)

    # print(outputFile, os.path.exists(outputFile))
    # print(clustersOutputFile, os.path.exists(clustersOutputFile))
    if os.path.exists(outputFile) and os.path.exists(clustersOutputFile):
        print("matlab skipped")
        return

    tempFile = dumpDataOnCleanCsv(dataLessTarget)

    if initialClusters is None:
        matlabKmeanCommand = "kmeans(csvread('{}'), {})".format(tempFile, str(clustersNumber))
    else:
        initialClustersMatlabStringMatrix = []
        for aClusterFeatures in initialClusters:
            initialClustersMatlabStringMatrix.append(",".join(map(lambda x: str(x), aClusterFeatures.tolist())))
        initialClustersMatlabStringMatrix = ";".join(initialClustersMatlabStringMatrix)

        matlabKmeanCommand = "kmeans(csvread('{}'), {}, 'Start', [{}])".format(tempFile, str(clustersNumber), initialClustersMatlabStringMatrix)

    command_parts = [MATLAB_EXE, "-nodisplay", "-nosplash", "-nodesktop", "-r \"rng('shuffle'); {}[idx,C] = {}; disp(idx); disp('===C==='); disp(num2str(C)); exit;\"   ".format(matlabRedirectTempFolder(TEMPFOLDER), matlabKmeanCommand)]
    print(" ".join(command_parts))
    result = subprocess.run(command_parts, stdout=subprocess.PIPE)
    res = result.stdout



    readingIdx = True

    i = 0
    resultFile = open(outputFile, 'w')
    for line in res.decode().split("\n"):
        if readingIdx and line == "===C===":
            resultFile.close()
            resultFile = open(clustersOutputFile, 'w')
            readingIdx = False
        else:
            if readingIdx:
                matches = re.fullmatch("     ?([0-9]+)", line)

                if matches is not None:
                    resultFile.write("{},{}\n".format(i, matches.group(1)))
                    i += 1
            else:
                if len(line.strip()) > 0:
                    resultFile.write(",".join(re.split(" +", line.strip())))
                    resultFile.write("\n")

    resultFile.close()


    os.unlink(tempFile)




# https://stackoverflow.com/questions/6685961/weka-simple-k-means-clustering-assignments
def wekaProcess(inFile, datasetName, runinfo = None):
    outputFile = datasetOutFile(datasetName, WEKA_ALGO, runinfo = runinfo)
    clustersOutputFile = datasetOutFile(datasetName, centroidFor(WEKA_ALGO), runinfo = runinfo)

    if os.path.exists(outputFile) and os.path.exists(clustersOutputFile):
        print("weka skipped")
        return

    command_parts = [JAVA_EXE, "-classpath", JAVA_CLASSPATH, "WekaRun", inFile, outputFile, clustersOutputFile]
    print(" ".join(command_parts))

    subprocess.call(command_parts)


def wekaUnormProcess(inFile, datasetName, runinfo = None):
    outputFile = datasetOutFile(datasetName, WEKA_UNORM_ALGO, runinfo = runinfo)
    clustersOutputFile = datasetOutFile(datasetName, centroidFor(WEKA_UNORM_ALGO), runinfo = runinfo)

    if os.path.exists(outputFile) and os.path.exists(clustersOutputFile):
        print("weka unorm skipped")
        return

    command_parts = [JAVA_EXE, "-classpath", JAVA_CLASSPATH, "WekaRunNorm", inFile, outputFile, clustersOutputFile]
    print(" ".join(command_parts))

    subprocess.call(command_parts)


# http://mlpack.org/man/kmeans.html
# http://www.mlpack.org/docs/mlpack-2.1.0/doxygen/formatdoc.html
def mlpackProcess(clustersNumber, dataLessTarget, datasetName, runinfo = None, initialClustersCsvFile = None):
    outputFile = datasetOutFile(datasetName, MLPACK_ALGO, runinfo=runinfo)
    clustersOutputFile = datasetOutFile(datasetName, centroidFor(MLPACK_ALGO), runinfo=runinfo)

    if os.path.exists(outputFile) and os.path.exists(clustersOutputFile):
        print("mlpack skipped")
        return

    tempFile = dumpDataOnCleanCsv(dataLessTarget)
    tempFile2 = "{}/{}b.csv".format(TEMPFOLDER, int(time.time()))


    # -a naive = Lloyd
    if initialClustersCsvFile is not None:
        command_parts = ["{}/mlpack_kmeans".format(MLPACK_BIN), "--clusters", str(clustersNumber), "-i", tempFile, "-I", initialClustersCsvFile, "-a", "naive", "-o", tempFile2, "-C", clustersOutputFile]
    else:
        command_parts = ["{}/mlpack_kmeans".format(MLPACK_BIN), "--clusters", str(clustersNumber), "-i", tempFile, "-o", tempFile2, "-a", "naive", "-C", clustersOutputFile]

    print(" ".join(command_parts))
    subprocess.call(command_parts)

    with open(tempFile2, 'r') as csvfile:
        with open(outputFile, 'w') as resultFile:
            i = 0
            resultReader = csv.reader(csvfile)
            for row in resultReader:
                resultFile.write("{},{}\n".format(i, int(float(row[-1]))))
                i += 1

    os.unlink(tempFile)
    os.unlink(tempFile2)

def sklearnProcess(clustersNumber, dataLessTarget, datasetName, runinfo = None, initialClusters = None, zeroTolerance = False):
    import sklearn.cluster

    selectedAlgo = SKLEARN_TOL0_ALGO if zeroTolerance else SKLEARN_ALGO
    outputFile = datasetOutFile(datasetName, selectedAlgo, runinfo=runinfo)
    clustersOutputFile = datasetOutFile(datasetName, centroidFor(selectedAlgo), runinfo=runinfo)

    print("File in {}".format(outputFile))
    print("Clusters in {}".format(clustersOutputFile))

    if os.path.exists(outputFile) and os.path.exists(clustersOutputFile):
        print("sklearn skipped")
        return


    # Create a KMean model.
    if initialClusters is None:
        # random_state is the seed to be used.
        # By default, k-means++ is used
        if zeroTolerance:
            sklearnKmeanModel = sklearn.cluster.KMeans(n_clusters=clustersNumber, tol=0)
        else:
            sklearnKmeanModel = sklearn.cluster.KMeans(n_clusters=clustersNumber)
    else:
        if zeroTolerance:
            sklearnKmeanModel = sklearn.cluster.KMeans(n_clusters=clustersNumber, init=initialClusters, tol=0)
        else:
            sklearnKmeanModel = sklearn.cluster.KMeans(n_clusters=clustersNumber, init=initialClusters)
        #sklearnKmeanModel = sklearn.cluster.KMeans(n_clusters=clustersNumber, init=initialClusters, tol = 0, algorithm = 'full', max_iter = 100)

    sklearnKmeanModel.fit(dataLessTarget)

    with open(outputFile, 'w') as csvfile:
        filewriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)

        for index, row in dataLessTarget.iterrows():
            filewriter.writerow([index, sklearnKmeanModel.labels_[index]])

    with open(clustersOutputFile, 'w') as clusterFile:
        filewriter = csv.writer(clusterFile, quoting=csv.QUOTE_MINIMAL)

        for row in sklearnKmeanModel.cluster_centers_:
            filewriter.writerow(row.tolist())




def shogunProcess(clustersNumber, dataLessTarget, datasetName, runinfo = None, initialClusters = None):
    import shogun

    outputFile = datasetOutFile(datasetName, SHOGUN_ALGO, runinfo=runinfo)
    clustersOutputFile = datasetOutFile(datasetName, centroidFor(SHOGUN_ALGO), runinfo=runinfo)

    if os.path.exists(outputFile) and os.path.exists(clustersOutputFile):
        print("shogun skipped")
        return

    train_features = shogun.RealFeatures(dataLessTarget.values.astype("float64").transpose())
    # distance metric over feature matrix - Euclidean distance
    distance = shogun.EuclideanDistance(train_features, train_features)

    # KMeans object created
    kmeans = shogun.KMeans(clustersNumber, distance)

    if initialClusters is None:
        # set KMeans++ flag
        kmeans.set_use_kmeanspp(True)
    else:
        # set new initial centers
        kmeans.set_initial_centers(initialClusters.astype("float64").transpose())

    # KMeans training
    kmeans.train()

    # cluster centers
    centers = kmeans.get_cluster_centers()

    # Labels for data points
    result = kmeans.apply()

    with open(outputFile, 'w') as csvfile:
        filewriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)

        for index, row in dataLessTarget.iterrows():
            filewriter.writerow([index, result[index].item(0)])

    with open(clustersOutputFile, 'w') as clusterFile:
        filewriter = csv.writer(clusterFile, quoting=csv.QUOTE_MINIMAL)

        for row in centers.transpose():
            filewriter.writerow(row.tolist())



def opencvProcess(clustersNumber, dataLessTarget, datasetName, runinfo = None, initialClusters = None):
    import cv2
    import numpy as np

    outputFile = datasetOutFile(datasetName, OPENCV_ALGO, runinfo=runinfo)
    clustersOutputFile = datasetOutFile(datasetName, centroidFor(OPENCV_ALGO), runinfo=runinfo)

    if os.path.exists(outputFile) and os.path.exists(clustersOutputFile):
        print("opencv skipped")
        return

    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0)

    if initialClusters is None:
        ret = cv2.kmeans(np.float32(dataLessTarget.values), clustersNumber, None, criteria, 10, flags=cv2.KMEANS_PP_CENTERS)
    else:
        print(initialClusters)
        ret = cv2.kmeans(np.float32(dataLessTarget.values), clustersNumber, initialClusters, criteria, 10, flags=cv2.KMEANS_USE_INITIAL_LABELS)

    with open(outputFile, 'w') as csvfile:
        filewriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)

        for index, row in dataLessTarget.iterrows():
            filewriter.writerow([index, ret[1][index].item(0)])

    with open(clustersOutputFile, 'w') as clusterFile:
        filewriter = csv.writer(clusterFile, quoting=csv.QUOTE_MINIMAL)

        for row in ret[2]:
            filewriter.writerow(row.tolist())



def rProcess(srcFile, datasetName, runinfo = None, initialClustersCsvFile = None, hundredIters = False):
    import pandas
    
    if initialClustersCsvFile is None and hundredIters:
        print("R kcca function don't have the iteration parameter !")
        return

    selectedAlgo = R_100ITER_ALGO if hundredIters else R_ALGO
    outputFile = datasetOutFile(datasetName, selectedAlgo, runinfo=runinfo)
    clustersOutputFile = datasetOutFile(datasetName, centroidFor(selectedAlgo), runinfo=runinfo)

    if os.path.exists(outputFile) and os.path.exists(clustersOutputFile):
        print("R skipped")
        return

    if initialClustersCsvFile is None:
        command_parts = [R_BIN, "--no-save", "--quiet", os.path.join(R_SCRIPT_BASE_DIR, "kmeans_test.R"), srcFile, outputFile, clustersOutputFile]
    else:
        command_parts = [R_BIN, "--no-save", "--quiet", os.path.join(R_SCRIPT_BASE_DIR, "kmeans_test_init_clusters_100it.R" if hundredIters else "kmeans_test_init_clusters.R"), srcFile, outputFile, clustersOutputFile, initialClustersCsvFile]

    subprocess.call(command_parts)

    print(" ".join(command_parts))
    dta = pandas.read_csv(clustersOutputFile)
    dta.drop(dta.columns[[0]], axis=1).to_csv(clustersOutputFile, index=False, header=False)



def tensorflowProcess(clustersNumber, dataLessTarget, datasetName, runinfo = None, initialClusters = None):
    import tensorflow as tf
    from numpy.core.tests.test_mem_overlap import xrange

    '''
    https://www.tensorflow.org/api_docs/python/tf/contrib/factorization/KMeansClustering
    '''
    outputFile = datasetOutFile(datasetName, TENSORFLOW_ALGO, runinfo=runinfo)
    clustersOutputFile = datasetOutFile(datasetName, centroidFor(TENSORFLOW_ALGO), runinfo=runinfo)

    if os.path.exists(outputFile) and os.path.exists(clustersOutputFile):
        print("tensorflow skipped")
        return

    points = dataLessTarget.values
    def input_fn():
        return tf.train.limit_epochs(tf.convert_to_tensor(points, dtype=tf.float32), num_epochs=1)

    if initialClusters is None:
        kmeans = tf.contrib.factorization.KMeansClustering(num_clusters=clustersNumber, use_mini_batch=False)
    else:
        kmeans = tf.contrib.factorization.KMeansClustering(num_clusters=clustersNumber, initial_clusters=initialClusters, use_mini_batch=False)

    # train
    num_iterations = 10
    previous_centers = None
    for _ in xrange(num_iterations):
        kmeans.train(input_fn)


    with open(outputFile, 'w') as csvfile:
        filewriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)

        cluster_indices = list(kmeans.predict_cluster_index(input_fn))
        for index, point in enumerate(points):
            cluster_index = cluster_indices[index]
            filewriter.writerow([index, cluster_index])

    # Clusters saving
    with open(clustersOutputFile, 'w') as clusterFile:
        filewriter = csv.writer(clusterFile, quoting=csv.QUOTE_MINIMAL)

        for row in kmeans.cluster_centers():
            filewriter.writerow(row.tolist())

    # with open(clustersOutputFile, 'w') as clusterFile:
    #     kmeans.cluster_centers().tofile(clusterFile)