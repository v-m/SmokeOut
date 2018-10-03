# Not used -- cf Xin for implementation
# Author: Vincenzo Musco (http://www.vmusco.com)
import csv
import hashlib
import os
import subprocess
import time

import sklearn.cluster

from config import TEMPFOLDER, MLPACK_BIN
from tools.static import SKLEARN_ALGO, datasetOutFile, centroidFor, MLPACK_ALGO


def computeMD5ResultIdentity(labels):
    md5 = hashlib.md5()

    for label in labels:
        md5.update(label)

    return md5.hexdigest()


def dumpDataOnCleanCsv(dataLessTarget):
    tempFile = "{}/{}.csv".format(TEMPFOLDER, int(time.time()))

    fp = open(tempFile, "w")
    fp.write(dataLessTarget.to_csv(index=False, header=False))
    fp.close()

    return tempFile



#http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
def sklearnProcess(dataLessTarget, datasetName, runinfo = None):
    selectedAlgo = SKLEARN_ALGO
    outputFile = datasetOutFile(datasetName, selectedAlgo, runinfo=runinfo)
    clustersOutputFile = datasetOutFile(datasetName, centroidFor(selectedAlgo), runinfo=runinfo)

    if os.path.exists(outputFile) and os.path.exists(clustersOutputFile):
        print("sklearn skipped")
        return


    # Create model.
    model = sklearn.cluster.DBSCAN()
    model.fit(dataLessTarget)

    with open(outputFile, 'w') as csvfile:
        filewriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)

        for index, row in dataLessTarget.iterrows():
            filewriter.writerow([index, model.labels_[index]])






# http://www.mlpack.org/docs/mlpack-3.0.2/man/mlpack_dbscan.html
def mlpackProcess(dataLessTarget, datasetName, runinfo = None):
    outputFile = datasetOutFile(datasetName, MLPACK_ALGO, runinfo=runinfo)
    clustersOutputFile = datasetOutFile(datasetName, centroidFor(MLPACK_ALGO), runinfo=runinfo)

    if os.path.exists(outputFile) and os.path.exists(clustersOutputFile):
        print("mlpack skipped")
        return

    tempFile = dumpDataOnCleanCsv(dataLessTarget)
    tempFile2 = "{}/{}b.csv".format(TEMPFOLDER, int(time.time()))

    command_parts = ["{}/mlpack_dbscan".format(MLPACK_BIN), "-i", tempFile, "-a", tempFile2]

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
