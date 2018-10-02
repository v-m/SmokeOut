import sys

from tools.static import *


def help():
    print("Please select an algorithm: {}".format(",".join([
        # AFFINITY_PROP_ALGO,
        # DBSCAN_ALGO,
        GAUSSIANMIX_ALGO,
        HIERARCHICAL_ALGO,
        KMEANS_ALGO,
        # MEANSHIFT_ALGO,
        # SPECTRAL_ALGO
        KMEANS_PLUSPLUS_ALGO
    ])))


if len(sys.argv) == 0:
    help()
else:
    print(sys.argv)
    theArg = sys.argv.pop(1)

    print(sys.argv)

    if theArg == GAUSSIANMIX_ALGO:
        import clustering.clustering_gaussianmix
    elif theArg == HIERARCHICAL_ALGO:
        import clustering.clustering_hierarchical
    elif theArg == KMEANS_ALGO:
        import clustering.clustering_kmeans_initrand
    elif theArg == KMEANS_PLUSPLUS_ALGO:
        import clustering.clustering_kmeans_kpp
    else:
        help()
