import sys

from tools.static import *


def help():
    print("Please select an algorithm:")
    print('\t%20s : %s'%(KMEANS_ALGO, "K-means (initial starting point)"))
    print('\t%20s : %s'%(KMEANS_PLUSPLUS_ALGO, "K-means++ (initial starting point)"))
    print('\t%20s : %s'%(HIERARCHICAL_ALGO, "Hierarchical Clustering"))
    print('\t%20s : %s'%(GAUSSIANMIX_ALGO, "Gaussian Mixtures"))


if len(sys.argv) < 2:
    help()
else:
    theArg = sys.argv.pop(1)

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
