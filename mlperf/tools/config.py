"""External toolkit paths"""

__author__ = "Vincenzo Musco (http://www.vmusco.com)"

import os


def initBinaryPath(bin_name, env_var_name):
    if env_var_name in os.environ:
        return os.environ[env_var_name]
    else:
        return bin_name


MLPACK_BIN = initBinaryPath("", "MLP_MLPACK_FOLDER")
MATLAB_EXE = initBinaryPath('matlab', "MLP_MATLAB_BIN")
OCTAVE_EXE = initBinaryPath('octave', "MLP_OCTAVE_BIN")
R_BIN = initBinaryPath('R', "MLP_R_BIN")
JAVA_EXE = initBinaryPath('java', "MLP_JAVA_BIN")
TEMPFOLDER = os.environ["MLP_TEMPFOLDER"] if "MLP_TEMPFOLDER" in os.environ else '/tmp'
