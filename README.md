# ml-perf

## Setup

This tool expect to get the binary path in two flavors:

- They are supplied in the system path
- They are passed with environement variables:

    | Variable | | Default |
    | --- | --- | --- |
    | `MLP_MLPACK_FOLDER` | The binary folder for mlpack installation |  |
    | `MLP_MATLAB_BIN` | The `matlab` binary | `matlab` |
    | `MLP_R_BIN` | The `RScript` binary | `RScript` |
    | `MLP_JAVA_BIN` | The `java` binary | `java` |
    | `MLP_TEMPFOLDER` | A temporary folder to work in | `/tmp` |

The latter may require you to write a short bash script for initializing the variables.