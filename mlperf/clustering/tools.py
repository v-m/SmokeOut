"""Tools used for clustering analysis"""

__author__ = "Vincenzo Musco (http://www.vmusco.com)"

import numpy
import os
import pandas

from mlperf.clustering.clusteringtoolkit import ClusteringToolkit


class DatasetFacts():
    """Object alternative to method read_dataset"""
    def __init__(self, data):
        self.data = data
        self.file_path = None

    def set_data(self, data):
        self.data = data

    def target(self):
        return self.data.target

    def ground_truth_cluster_ids(self):
        return self.target().unique()

    def number_clusters(self):
        return len(self.ground_truth_cluster_ids())

    def data_without_target(self):
        return self.data.loc[:, self.data.columns != 'target']

    def nb_instances(self):
        """number of instances"""
        return self.data.shape[0]

    def nb_features(self):
        """number of features (excluding target)"""
        return self.data.shape[1] - 1

    @staticmethod
    def read_dataset(source_file):
        print("Reading file {}...".format(source_file))
        # data = pandas.read_csv(srcFile, sep='\t')
        # data = dd.read_csv(srcFile, sep='\t')

        chunksize = 100000
        text_file_reader = pandas.read_csv(source_file, sep='\t', chunksize=chunksize, iterator=True)
        data = pandas.concat(text_file_reader, ignore_index=True)

        ret = DatasetFacts(data)
        ret.file_path = source_file
        return ret


def run_for_nr(run_base, run_id):
    return "{}{}".format(run_base, run_id)


def read_dataset(source_file):
    print("Reading file {}...".format(source_file))
    # data = pandas.read_csv(srcFile, sep='\t')
    # data = dd.read_csv(srcFile, sep='\t')

    chunksize = 100000
    text_file_reader = pandas.read_csv(source_file, sep='\t', chunksize=chunksize, iterator=True)
    data = pandas.concat(text_file_reader, ignore_index=True)

    print("Analyzing file...")
    ground_truth_cluster_ids = data.target.unique()
    number_clusters = len(ground_truth_cluster_ids)
    print("#clusters = {}".format(number_clusters))

    data_without_target = data.loc[:, data.columns != 'target']

    return {
        'data': data,
        'data_without_target': data_without_target,
        'number_clusters': number_clusters,
        'target': data.target,
        'ground_truth_cluster_ids': ground_truth_cluster_ids
    }


def read_centroids_file(drawn_clusters_file_path):
    return pandas.read_csv(drawn_clusters_file_path, header=None, dtype='float32').values


def draw_centroids(ground_truth_clusters_id, data, drawn_clusters_file_path=None):
    initial_clusters = list()

    for i in ground_truth_clusters_id:
        found = False
        selected_sample = None

        '''
         In some dataset (eg. titanic) the random drawn cluster centroid may be the same in both clusters. To 
            avoid this effect, we redrawn as long as there is a conflict...
        '''
        while not found:
            selected_sample = data[data.target == i].sample(1)
            selected_sample = selected_sample.loc[:, data.columns != 'target'].iloc[0].values

            found = True

            for anInitialClusterPreviouslyInserted in initial_clusters:
                if False not in (selected_sample == anInitialClusterPreviouslyInserted):
                    found = False
                    break

        initial_clusters.append(selected_sample)

    initial_clusters = numpy.asarray(initial_clusters)
    if drawn_clusters_file_path:
        pandas.DataFrame(initial_clusters).to_csv(path_or_buf=drawn_clusters_file_path, index=False, header=False)

    return initial_clusters


def read_or_draw_centroids(dataset_name, run_info, ground_truth_clusters_id, data):
    drawn_clusters_file_path = ClusteringToolkit.dataset_out_file_name_static(dataset_name,
                                                                              "{}.init_set_clusters".format(run_info))

    if not os.path.exists(drawn_clusters_file_path):
        # Lets draw a random feature set on EACH feature (this will be the starting point for *ALL* algorithms)
        initial_clusters = draw_centroids(ground_truth_clusters_id, data, drawn_clusters_file_path)
    else:
        # Reread to get float32 type (required by TF)
        initial_clusters = read_centroids_file(drawn_clusters_file_path)

    return drawn_clusters_file_path, initial_clusters
