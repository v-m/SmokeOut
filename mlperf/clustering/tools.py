#
# Author: Vincenzo Musco (http://www.vmusco.com)

import hashlib
import time
import random

from mlperf.tools.config import TEMPFOLDER

from scipy.stats.mstats import gmean
from numpy import array

def computeMD5ResultIdentity(labels):
    md5 = hashlib.md5()

    for label in labels:
        md5.update(label)

    return md5.hexdigest()


def dumpDataOnCleanCsv(dataLessTarget):
    tempFile = "{}/{}_{}.csv".format(TEMPFOLDER, int(time.time()), random.randint(1,10000))

    fp = open(tempFile, "w")
    fp.write(dataLessTarget.to_csv(index=False, header=False))
    fp.close()

    return tempFile


def gmeanFixed(alist):
    gmeanConverted = list(map(lambda x: x + 1.1, alist))
    ret = gmean(array(gmeanConverted)) - 1.1
    return float(ret)
