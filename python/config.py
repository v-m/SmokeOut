#
# Author: Vincenzo Musco (http://www.vmusco.com)

MLPACK_BIN = "/Users/vince/System/Software/mlpack-3.0.1/build/bin"

MATLAB_EXE = "/Applications/MATLAB_R2017b.app/bin/matlab"
#Thelma: MATLAB_EXE = "/afs/cad.njit.edu/sw.common/matlab-2018a/bin/matlab"

R_BIN = "/usr/local/bin/RScript"
#Thelma: R_BIN = "/afs/cad/linux/R-3.5.0/bin/Rscript"

JAVA_EXE = "/Library/Java/JavaVirtualMachines/jdk-9.0.4.jdk/Contents/Home/bin/java"
#Thelma: JAVA_EXE = "/afs/cad/linux/java8/bin/java"

SKIP_DATASET = ["connect-4", "poker", "mnist", "kddcup"]

TEMPFOLDER = "/tmp"
DATASET_ROOT = "/Users/vince/Temp/Datasets/learning_dataset/penn-ml-benchmarks/datasets/classification"
# DATASET_ROOT = "/tmp/penn-ml-benchmarks/datasets/classification"
# DATASET_ROOT = "/Users/vince/Temp/Datasets/temp-pennml/datasets/classification"

# DATASET_ROOT_ALL = ["/Users/vince/Temp/Datasets/penn-ml-benchmarks/datasets/classification",
#                     "/Users/vince/Temp/Datasets/penn-ml-benchmarks_old/datasets/classification"]

DATASET_ROOT_ALL = ["/Users/vince/Temp/Datasets/learning_dataset/penn-ml-benchmarks/datasets/classification",
                    "/Users/vince/Temp/Datasets/learning_dataset/penn-ml-benchmarks-xin/datasets/classification"]

# DATASET_ROOT_ALL = [DATASET_ROOT]

NB_RUNS = 10
