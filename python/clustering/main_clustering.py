#
# Author: Vincenzo Musco (http://www.vmusco.com)
import argparse
import sys
import pandas
# import dask.dataframe as dd
from tools.static import *

class ClusterPipeline:
    def __init__(self, RUN_INFO_BASE, AVAIL_ALGOS, pkg):
        self.RUN_INFO_BASE = RUN_INFO_BASE
        self.AVAIL_ALGOS = AVAIL_ALGOS
        self.pkg = pkg

    def preprocessRun(self, groundTruthClustersId, data, dataLessTarget, datasetName, RUN_INFO):
        pass

    def processRun(self, ALGO, srcFile, clustersNumber, dataLessTarget, datasetName, RUN_INFO):
        run = self.pkg

        # SHOGUN
        if ALGO == SHOGUN_ALGO:
            run.shogunProcess(clustersNumber, dataLessTarget, datasetName, RUN_INFO)

        # SCIKIT
        elif ALGO == SKLEARN_ALGO:
            run.sklearnProcess(clustersNumber, dataLessTarget, datasetName, RUN_INFO)

        # R
        elif ALGO == R_ALGO:
            run.rProcess(srcFile, datasetName, RUN_INFO)

        # Tensorflow
        elif ALGO == TENSORFLOW_ALGO:
            run.tensorflowProcess(clustersNumber, dataLessTarget, datasetName, RUN_INFO)

        # MLPack
        elif ALGO == MLPACK_ALGO:
            run.mlpackProcess(clustersNumber, dataLessTarget, datasetName, RUN_INFO)

        # Matlab
        elif ALGO == MATLAB_ALGO:
            run.matlabProcess(clustersNumber, dataLessTarget, datasetName, RUN_INFO)

        # Weka
        elif ALGO == WEKA_ALGO:
            run.wekaProcess(srcFile, datasetName, RUN_INFO)

        # OPENCV
        elif ALGO == OPENCV_ALGO:
            run.opencvProcess(clustersNumber, dataLessTarget, datasetName, RUN_INFO)

        else:
            self.otherProcessRun(ALGO, srcFile, clustersNumber, dataLessTarget, datasetName, RUN_INFO)

    def runPipe(self):
        parser = argparse.ArgumentParser(description='Generate clusters for dataset')
        parser.add_argument('--toolkits', '-t', action='append', help='Consider these toolkits', default=None)
        parser.add_argument('--base', '-b', type=int, action='append', help='Execute only RUN x', default=0)
        parser.add_argument('--runs', '-r', type=int, help='Number of runs to perform', default=NB_RUNS)
        parser.add_argument('--dataset', '-d', action='append', help='Run this dataset tsv file', default=None)
        args = parser.parse_args()

        CONSIDERED_ALGOS = self.AVAIL_ALGOS if args.toolkits is None else args.toolkits
        print("Algos = {}".format(CONSIDERED_ALGOS))
        print("Runs = {}".format(args.runs))
        print("Dataset = {}".format(args.dataset))
        print("Base = {}".format(args.base))

        for runid in range(args.runs):
            RUN_INFO = runForNr(self.RUN_INFO_BASE, args.base + runid)
            print("*****")
            print("RUN: {}".format(RUN_INFO))
            print("*****")

            for dataset in args.dataset:
                print("Considering: {}".format(CONSIDERED_ALGOS))

                srcFile = dataset

                print("Reading file {}...".format(srcFile))
                # data = pandas.read_csv(srcFile, sep='\t')
                # data = dd.read_csv(srcFile, sep='\t')

                chunksize = 100000
                text_file_reader = pandas.read_csv(srcFile, sep='\t', chunksize=chunksize, iterator=True)
                data = pandas.concat(text_file_reader, ignore_index=True)

                ''' VINCE 
                We know that the dataframe contains a target column with the expected class.
                Let's see what values are contained withit this column
                '''
                print("Analyzing file...".format(srcFile))
                groundTruthClustersId = data.target.unique()
                clustersNumber = len(groundTruthClustersId)

                dataLessTarget = data.loc[:, data.columns != 'target']

                self.preprocessRun(groundTruthClustersId, data, dataLessTarget, srcFile, RUN_INFO)

                for ALGO in CONSIDERED_ALGOS:
                    print("RUN {}. ALGO = {}".format(runid, ALGO))
                    try:
                        self.processRun(ALGO, srcFile, clustersNumber, dataLessTarget, srcFile, RUN_INFO)
                    except:
                        print("!!!!! Exception for {} !!!!!".format(ALGO))
                        print(sys.exc_info())

    def otherProcessRun(self, ALGO, srcFile, clustersNumber, dataLessTarget, datasetName, RUN_INFO):
        pass