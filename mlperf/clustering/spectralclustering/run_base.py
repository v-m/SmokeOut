import csv
import os
import re
import subprocess

from mlperf.clustering.tools import dumpDataOnCleanCsv
from mlperf.tools.config import MATLAB_EXE, TEMPFOLDER, JAVA_EXE, R_BIN
from mlperf.tools.static import datasetOutFile, MATLAB_ALGO, matlabRedirectTempFolder, WEKA_ALGO, JAVA_CLASSPATH, \
    SKLEARN_ALGO, R_ALGO, SHOGUN_ALGO

def sklearnProcess(clustersNumber, dataLessTarget, datasetName, runinfo = None):
    import sklearn.cluster
    
    selectedAlgo = SKLEARN_ALGO
    outputFile = datasetOutFile(datasetName, selectedAlgo, runinfo=runinfo)

    if os.path.exists(outputFile):
        print("sklearn skipped")
        return

    check = False

    while not check:
        try:
            builtModel = sklearn.cluster.SpectralClustering(n_clusters=clustersNumber)
            builtModel.fit(dataLessTarget)
            check = True
        except np.linalg.linalg.LinAlgError:
            continue
        except AssertionError:
            continue

    with open(outputFile, 'w') as csvfile:
        filewriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)

        for index, row in dataLessTarget.iterrows():
            filewriter.writerow([index, builtModel.labels_[index]])
