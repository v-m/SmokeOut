/*
# Simple script for running Weka k-means (normalized)
# Author: Vincenzo Musco (http://www.vmusco.com)
*/

import weka.clusterers.SimpleKMeans;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.CSVLoader;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.HashSet;
import java.util.zip.GZIPInputStream;

public class WekaRun {
    public static void main(String[] args) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setFieldSeparator("\t");
        loader.setSource(new GZIPInputStream(new FileInputStream(args[0])));

        SimpleKMeans kmeans = new SimpleKMeans();

        // This is the important parameter to set
        kmeans.setPreserveInstancesOrder(true);


        if(args.length > 3){
            String[] argsParts = args[4].split(";");
            for(String part : argsParts) {
                String[] subparts = part.split("=");

                if (subparts[0].equals("nbiter")) {
                    kmeans.setMaxIterations(Integer.parseInt(subparts[1]));
                }else if(subparts[0].equals("unorm")) {
                    EuclideanDistance dist = new EuclideanDistance();
                    dist.setDontNormalize(true);
                    kmeans.setDistanceFunction(dist);
                }else if(subparts[0].equals("seed")) {
                    kmeans.setSeed(Integer.parseInt(subparts[1]));
                }
            }
        }


        Instances dataset = loader.getDataSet();

        HashSet<Double> classes = new HashSet<>();
        for(int i = 0; i<dataset.numInstances(); i++){
            classes.add(dataset.instance(i).value(dataset.attribute("target")));
        }

        kmeans.setNumClusters(classes.size());
        kmeans.setInitializationMethod(new SelectedTag(SimpleKMeans.KMEANS_PLUS_PLUS, SimpleKMeans.TAGS_SELECTION));
        classes.clear();

        dataset.deleteAttributeAt(dataset.attribute("target").index());
        System.out.println(dataset);
        kmeans.buildClusterer(dataset);

        // This array returns the cluster number (starting with 0) for each instance
        // The array has as many elements as the number of instances
        int[] assignments = kmeans.getAssignments();

        File f = new File(args[1]);
        if(f.exists())
            f.delete();

        File fcentroids = new File(args[2]);
        if(fcentroids.exists())
            fcentroids.delete();

        FileOutputStream fos = new FileOutputStream(f);
        int i=0;
        for(int clusterNum : assignments) {
            fos.write(String.format("%d,%d\n", i, clusterNum).getBytes());
            i++;
        }
        fos.close();

        fos = new FileOutputStream(fcentroids);
        for(Instance centroid : kmeans.getClusterCentroids()) {
            fos.write(centroid.toString().getBytes());
            fos.write("\n".getBytes());
        }
        fos.close();
    }
}
