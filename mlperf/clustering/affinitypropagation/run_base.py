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

    #print(clustersNumber, dataLessTarget, datasetName, runinfo)
    i = re.fullmatch("[^0-9]*?([0-9]+)", runinfo)
    i = int(i.group(1))

    damping_value = 0.016*i + 0.5
    builtModel = sklearn.cluster.AffinityPropagation(damping = damping_value)
    builtModel.fit(dataLessTarget)

    with open(outputFile, 'w') as csvfile:
        filewriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)

        for index, row in dataLessTarget.iterrows():
            filewriter.writerow([index, builtModel.labels_[index]])
