# Entry point for generating K-means (random starting points)
# Author: Vincenzo Musco (http://www.vmusco.com)
import os
import numpy
import pandas
import mlperf.clustering.kmeans.run_base as run
from mlperf.clustering.main_clustering import ClusterPipeline


from mlperf.tools.static import INCLUDED_ALGO, KMEANS_ALGO, datasetOutFile, SHOGUN_ALGO, SKLEARN_ALGO, R_ALGO, \
    TENSORFLOW_ALGO, MLPACK_ALGO, MATLAB_ALGO, SKLEARN_TOL0_ALGO, R_100ITER_ALGO

RUN_INFO_BASE = KMEANS_ALGO
AVAIL_ALGOS = INCLUDED_ALGO[RUN_INFO_BASE]


class KMeansFixedStartingPoint(ClusterPipeline):
    def __init__(self):
        super().__init__(RUN_INFO_BASE, AVAIL_ALGOS, run)
        self.initialClusters = None
        self.drawnInitialClustersFeatures = None

    def preprocessRun(self, groundTruthClustersId, data, dataLessTarget, datasetName, RUN_INFO):
        self.drawnInitialClustersFeatures = datasetOutFile(datasetName, "{}.init_set_clusters".format(RUN_INFO))

        if os.path.exists(self.drawnInitialClustersFeatures):
            print("Loading selected clusters")
            self.initialClusters = pandas.read_csv(self.drawnInitialClustersFeatures, header=None,
                                                   dtype='float32').values
        else:
            # At this stage, we draw one random feature set on EACH feature (this will be the starting point for *ALL* algorithms)
            self.initialClusters = list()

            for i in groundTruthClustersId:
                found = False
                selectedSample = None

                '''
                 In some dataset (eg. titanic) the random drawn cluster centroid may be the same in both clusters. To 
                    avoid this effect, we redrawn as long as there is a conflict...
                '''
                while not found:
                    selectedSample = data[data.target == i].sample(1)
                    selectedSample = selectedSample.loc[:, data.columns != 'target'].iloc[0].values

                    found = True

                    for anInitialClusterPreviouslyInserted in self.initialClusters:
                        if False not in (selectedSample == anInitialClusterPreviouslyInserted):
                            found = False
                            break

                self.initialClusters.append(selectedSample)

            self.initialClusters = numpy.asarray(self.initialClusters)
            print("Saving initial clusters for this project...")
            pandas.DataFrame(self.initialClusters).to_csv(path_or_buf=self.drawnInitialClustersFeatures, index=False,
                                                          header=False)
            # Reread to get float32 type (required by TF)
            self.initialClusters = pandas.read_csv(self.drawnInitialClustersFeatures, header=None,
                                                   dtype='float32').values

    def processRun(self, ALGO, srcFile, clustersNumber, dataLessTarget, datasetName, RUN_INFO):
        run = self.pkg

        # SHOGUN
        if ALGO == SHOGUN_ALGO:
            run.shogunProcess(clustersNumber, dataLessTarget, datasetName, RUN_INFO, self.initialClusters)

        # SCIKIT
        elif ALGO == SKLEARN_ALGO:
            run.sklearnProcess(clustersNumber, dataLessTarget, datasetName, RUN_INFO, self.initialClusters)

        # R
        elif ALGO == R_ALGO:
            run.rProcess(srcFile, datasetName, RUN_INFO, self.drawnInitialClustersFeatures)

        # Tensorflow
        elif ALGO == TENSORFLOW_ALGO:
            run.tensorflowProcess(clustersNumber, dataLessTarget, datasetName, RUN_INFO, self.initialClusters)

        # MLPack
        elif ALGO == MLPACK_ALGO:
            run.mlpackProcess(clustersNumber, dataLessTarget, datasetName, RUN_INFO, self.drawnInitialClustersFeatures)

        # Matlab
        elif ALGO == MATLAB_ALGO:
            run.matlabProcess(clustersNumber, dataLessTarget, datasetName, RUN_INFO, self.initialClusters)

        else:
            self.otherProcessRun(ALGO, srcFile, clustersNumber, dataLessTarget, datasetName, RUN_INFO)

    def otherProcessRun(self, ALGO, srcFile, clustersNumber, dataLessTarget, datasetName, RUN_INFO):
        # SCIKIT
        if ALGO == SKLEARN_TOL0_ALGO:
            run.sklearnProcess(clustersNumber, dataLessTarget, datasetName, RUN_INFO, self.initialClusters,
                               zeroTolerance=True)

        # R
        if ALGO == R_100ITER_ALGO:
            run.rProcess(srcFile, datasetName, RUN_INFO, self.drawnInitialClustersFeatures, hundredIters=True)


KMeansFixedStartingPoint().runPipe()
