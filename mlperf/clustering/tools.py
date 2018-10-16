"""Tools used for clustering analysis"""

__author__ = "Vincenzo Musco (http://www.vmusco.com)"

import numpy
import os
import pandas

from mlperf.clustering.clusteringtoolkit import ClusteringToolkit


def run_for_nr(run_base, run_id):
    return "{}{}".format(run_base, run_id)


def draw_centroids(datasetName, run_info, ground_truth_clusters_id, data):
    drawn_clusters_file_path = ClusteringToolkit.dataset_out_file_name_static(datasetName,
                                                                    "{}.init_set_clusters".format(run_info))

    if not os.path.exists(drawn_clusters_file_path):
        # Lets draw a random feature set on EACH feature (this will be the starting point for *ALL* algorithms)
        initial_clusters = list()

        for i in ground_truth_clusters_id:
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

                for anInitialClusterPreviouslyInserted in initial_clusters:
                    if False not in (selectedSample == anInitialClusterPreviouslyInserted):
                        found = False
                        break

            initial_clusters.append(selectedSample)

        initial_clusters = numpy.asarray(initial_clusters)
        pandas.DataFrame(initial_clusters).to_csv(path_or_buf=drawn_clusters_file_path, index=False, header=False)
    else:
        # Reread to get float32 type (required by TF)
        initial_clusters = pandas.read_csv(drawn_clusters_file_path, header=None, dtype='float32').values

    return drawn_clusters_file_path, initial_clusters
