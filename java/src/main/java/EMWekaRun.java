/*
# Simple script for running Weka EM
# Author: Vincenzo Musco (http://www.vmusco.com)
*/

import weka.clusterers.EM;
import weka.clusterers.HierarchicalClusterer;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.HashSet;
import java.util.zip.GZIPInputStream;

public class EMWekaRun {
    public static void main(String[] args) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setFieldSeparator("\t");
        loader.setSource(new GZIPInputStream(new FileInputStream(args[0])));

        EM model = new EM();
        if(args.length > 2){
            String[] argsParts = args[2].split(";");
            for(String part : argsParts) {
                String[] subparts = part.split("=");

                if(subparts[0].equals("seed")) {
                    model.setSeed(Integer.parseInt(subparts[1]));
                }
            }
        }

        // This is the important parameter to set
        Instances dataset = loader.getDataSet();

        HashSet<Double> classes = new HashSet<>();
        for(int i = 0; i<dataset.numInstances(); i++){
            classes.add(dataset.instance(i).value(dataset.attribute("target")));
        }

        model.setNumClusters(classes.size());

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
