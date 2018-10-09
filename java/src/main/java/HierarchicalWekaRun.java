/*
# Simple script for running Weka hierarchical clustering
# Author: Vincenzo Musco (http://www.vmusco.com)
*/

import weka.clusterers.HierarchicalClusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.*;
import weka.core.converters.CSVLoader;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.HashSet;
import java.util.zip.GZIPInputStream;

public class HierarchicalWekaRun {
    public static void main(String[] args) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setFieldSeparator("\t");
        loader.setSource(new GZIPInputStream(new FileInputStream(args[0])));


        HierarchicalClusterer model = new HierarchicalClusterer();

        // This is the important parameter to set
        Instances dataset = loader.getDataSet();

        HashSet<Double> classes = new HashSet<>();
        for(int i = 0; i<dataset.numInstances(); i++){
            classes.add(dataset.instance(i).value(dataset.attribute("target")));
        }

        model.setNumClusters(classes.size());
        model.setOptions(new String[]{ "-L", "WARD" });
//        Euclidian Default!
//        model.setDistanceFunction(new EuclideanDistance());

        classes.clear();

        dataset.deleteAttributeAt(dataset.attribute("target").index());
        model.buildClusterer(dataset);

        File f = new File(args[1]);
        if(f.exists())
            f.delete();

        FileOutputStream fos = new FileOutputStream(f);
        for(int i = 0; i<dataset.numInstances(); i++){
            Instance thisInstance = dataset.instance(i);
            int clusterSet = model.clusterInstance(thisInstance);
            fos.write(String.format("%d,%d\n", i, clusterSet).getBytes());
        }
        fos.close();
    }
}
