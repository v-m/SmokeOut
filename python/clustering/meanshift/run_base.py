import csv
import os
import subprocess
import time
import sklearn.cluster

from clustering.tools import dumpDataOnCleanCsv


#ALGONAME = "meanshift_"
from config import TEMPFOLDER, MLPACK_BIN
from tools.static import datasetOutFile, centroidFor

ALGONAME = ""
MLPACK_ALGO = "{}mlpack".format(ALGONAME)
SKLEARN_ALGO = "{}sklearn".format(ALGONAME)

# http://www.mlpack.org/docs/mlpack-3.0.1/man/mlpack_mean_shift.html
def mlpackProcess(dataLessTarget, datasetName, runinfo = None):
    outputFile = datasetOutFile(datasetName, MLPACK_ALGO, runinfo=runinfo)
    clustersOutputFile = datasetOutFile(datasetName, centroidFor(MLPACK_ALGO), runinfo=runinfo)

    if os.path.exists(outputFile) and os.path.exists(clustersOutputFile):
        print("mlpack skipped")
        return

    tempFile = dumpDataOnCleanCsv(dataLessTarget)
    tempFile2 = "{}/{}b.csv".format(TEMPFOLDER, int(time.time()))

    command_parts = ["{}/mlpack_mean_shift".format(MLPACK_BIN), "-i", tempFile, "-o", tempFile2]

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

def sklearnProcess(dataLessTarget, datasetName, runinfo = None):
    selectedAlgo = SKLEARN_ALGO
    outputFile = datasetOutFile(datasetName, selectedAlgo, runinfo=runinfo)
    clustersOutputFile = datasetOutFile(datasetName, centroidFor(selectedAlgo), runinfo=runinfo)

    if os.path.exists(outputFile) and os.path.exists(clustersOutputFile):
        print("sklearn skipped")
        return


    # Create model.
    model = sklearn.cluster.MeanShift()
    model.fit(dataLessTarget)

    with open(outputFile, 'w') as csvfile:
        filewriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)

        for index, row in dataLessTarget.iterrows():
            filewriter.writerow([index, model.labels_[index]])

    with open(clustersOutputFile, 'w') as clusterFile:
        filewriter = csv.writer(clusterFile, quoting=csv.QUOTE_MINIMAL)

        for row in model.cluster_centers_:
            filewriter.writerow(row.tolist())

