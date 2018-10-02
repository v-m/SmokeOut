#
# Author: Vincenzo Musco (http://www.vmusco.com)
import argparse
import sys
import pandas
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
        parser.add_argument('--algos', '-A', action='append', help='Consider these algorithms', default=None)
        parser.add_argument('--base', '-B', type=int, action='append', help='Execute only RUN x', default=0)
        parser.add_argument('--runs', '-R', type=int, help='Number of runs to perform', default=NB_RUNS)
        parser.add_argument('--dataset', '-D', action='append', help='Execute only dataset X', default=None)
        args = parser.parse_args()

        CONSIDERED_ALGOS = self.AVAIL_ALGOS if args.algos is None else args.algos
        print("Algos = {}".format(CONSIDERED_ALGOS))
        print("Runs = {}".format(args.runs))
        print("Dataset = {}".format(args.dataset))
        print("Base = {}".format(args.base))

        # BASE = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        # DATASETFILTER = sys.argv[3] if len(sys.argv) > 3 else None
        for runid in range(args.runs):

            RUN_INFO = runForNr(self.RUN_INFO_BASE, args.base + runid)
            print("*****")
            print("RUN: {}".format(RUN_INFO))
            print("*****")

            for datasetName in exploreDatasets():
                if args.dataset is not None and datasetName not in args.dataset:
                    continue

                print(datasetName)

                if datasetName in SKIP_DATASET:
                    print("Skipped !")
                    continue

                # if len(sys.argv) > 1:
                #     CONSIDERED_ALGOS = sys.argv[1].split(":")
                print("Considering: {}".format(CONSIDERED_ALGOS))

                srcFile = datasetSrcFile(datasetName)

                if not os.path.exists(srcFile):
                    continue

                data = pandas.read_csv(srcFile, sep='\t')

                ''' VINCE 
                We know that the dataframe contains a target column with the expected class.
                Let's see what values are contained withit this column
                '''
                groundTruthClustersId = data.target.unique()
                clustersNumber = len(groundTruthClustersId)

                dataLessTarget = data.loc[:, data.columns != 'target']

                self.preprocessRun(groundTruthClustersId, data, dataLessTarget, datasetName, RUN_INFO)

                for ALGO in CONSIDERED_ALGOS:
                    print("RUN {}. ALGO = {}".format(runid, ALGO))
                    try:
                        self.processRun(ALGO, srcFile, clustersNumber, dataLessTarget, datasetName, RUN_INFO)
                    except:
                        print("!!!!! Exception for {} !!!!!".format(ALGO))
                        print(sys.exc_info())

    def otherProcessRun(self, ALGO, srcFile, clustersNumber, dataLessTarget, datasetName, RUN_INFO):
        pass