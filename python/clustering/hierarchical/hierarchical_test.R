# Title     : TODO
# Objective : TODO
# Created by: vince
# Created on: 5/14/18

library(cluster)

args = commandArgs(trailingOnly=TRUE)

# Load a GZ file
zz=gzfile(args[1])
# Parse it as a CSV file
dat=read.csv(zz,header=T, sep='\t')

# Let's remove the target feature
clustersNumber = nrow(unique(dat["target"]))
datWithoutTarget = subset( dat, select = -target )

d <- dist(datWithoutTarget, method = "euclidean")
hc1 <- hclust(d, method = "ward.D2" )
sub_grp <- cutree(hc1, k = clustersNumber)

write.csv(sub_grp, file=args[2])