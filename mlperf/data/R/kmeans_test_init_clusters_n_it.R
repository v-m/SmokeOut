# Created by: Vincenzo Musco (http://www.vmusco.com)
# Created on: 2018-10-16

args = commandArgs(trailingOnly=TRUE)

# Load a GZ file
zz=gzfile(args[1])
# Parse it as a CSV file
dat=read.csv(zz,header=T, sep='\t')

# Let's remove the target feature
clustersNumber = nrow(unique(dat["target"]))
datWithoutTarget = subset( dat, select = -target )

# http://stat.ethz.ch/R-manual/R-devel/library/stats/html/kmeans.html
init_clusters = read.csv(args[4], header = FALSE)

clusteringResult = kmeans(datWithoutTarget, init_clusters, iter.max = args[4], algorithm='Lloyd') #, nstart = 2,
write.csv(clusteringResult["cluster"], file=args[2])
write.csv(clusteringResult["centers"], file=args[3])

#clusters = clusteringResult$cluster
#print(clusteringResult)
#print(clusters(clusteringResult))

# [1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
# [6] "betweenss"    "size"         "iter"         "ifault"
