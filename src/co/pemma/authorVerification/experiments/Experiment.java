package co.pemma.authorVerification.experiments;

import co.pemma.authorVerification.*;
import co.pemma.authorVerification.NGramVectors.TokenType;
import co.pemma.authorVerification.cluster.Clusterer;
import co.pemma.authorVerification.cluster.ImpostorCluster;
import co.pemma.authorVerification.cluster.NGramBaseLineClustering;
import co.pemma.authorVerification.cluster.WritePrintsBaseLineClustering;
import org.apache.mahout.common.distance.CosineDistanceMeasure;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Vector;
import org.javatuples.Pair;
import org.javatuples.Triplet;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

public abstract class Experiment
{
    // location of input data
    protected static String DATA_LOCATION;
    // max number of imposters to randomly select from input
    public static int TOTAL_IMPOSTERS = 250;
    public int testSize = 250;
    public int trainSize = 0;

    double[] sameAuthor;
    double[] diffAuthor;

    List<String> impostors;
    Random rand;
    int bins;
    boolean printSimDist = false;
    double threshold;
    double merge;

    public Experiment()
    {
        this.rand 					= new Random();
        this.bins					= Parameters.BINS;
        sameAuthor 					= new double[bins];
        diffAuthor 					= new double[bins];
        reset();
    }

    protected void reset()
    {
        for (int i = 0; i < bins; i++)
        {
            sameAuthor[i] = 0;
            diffAuthor[i] = 0;
        }
        this.impostors = new ArrayList<String>();
    }

    protected abstract void pairs(String output, TokenType feature);

    protected void learnPairs(String output, String classifierType, boolean ngrams, boolean tfidf, boolean writeprints){}

    protected abstract void mistakenImpostors(String output, int mistakenImpostors, int divisions);

    /**
     * trains pairs with optional features for ngram min-max similarity, tfidf minmax similarity, and writeprints cosine similarity
     */
    protected ArrayList<Attribute> trainPairs(List<Triplet<String, String, Integer>> trainPairs, Map<String, Vector> ngramVectors, Map<String, Vector> tfidfVectors, Map<String, Vector> writeprintsVectors,
                                              boolean ngrams, boolean tfidf, boolean writeprints, Map<String, String> postData, Classifier classifier)
    {
        System.out.println("Setting up training pair data using " + trainSize + " authors...");

        // train the classifier model
        System.out.println("Training regression model...");
        ArrayList<Attribute> features = new ArrayList<Attribute>();
        features.add(new Attribute("impostor"));
        List<String> classLabels = new ArrayList<String>();
        classLabels.add("1");
        classLabels.add("0");

        if (ngrams)
            features.add(new Attribute("ngrams"));
        if (tfidf)
        {
            tfidfVectors = NGramVectors.run(postData, TokenType.TERMS);
            features.add(new Attribute("tfidf"));
        }
        if (writeprints)
        {
            writeprintsVectors = WritePrintsVectors.jstyloVectors(postData, new ArrayList<>(postData.keySet()));
            features.add(new Attribute("writeprints"));
        }
        features.add(new Attribute("class", classLabels));

        Instances trainSet = new Instances("train", features, trainPairs.size());
        trainSet.setClassIndex(trainSet.numAttributes()-1);
        for (Triplet<String, String, Integer> pair : trainPairs)
        {
            Instance in = similarityVector("f,"+pair.getValue0(), "l,"+pair.getValue1(), features, ngrams, ngramVectors, tfidfVectors, writeprintsVectors);
            if (in != null)
            {
                in.setDataset(trainSet);
                in.setClassValue(""+pair.getValue2());
                trainSet.add(in);
            }
            //			Utilities.printPercentProgress(i, trainPairs.size());
        }
        classifier.batchTrain(trainSet);
//        return null;
        return features;
    }

    /**
     * trains pairs with optional features for ngram min-max similarity, tfidf minmax similarity, and writeprints cosine similarity
     */
    protected ArrayList<Attribute> trainPairs(List<Pair<String, String>> trainPairs, int[] trainLabels,
                                              Map<String, Vector> ngramVectors, Map<String, Vector> tfidfVectors, Map<String, Vector> writeprintsVectors,
                                              boolean ngrams, boolean tfidf, boolean writeprints, Map<String, String> postData, Classifier classifier)
    {
        System.out.println("Setting up training pair data using " + trainSize + " authors...");

        // train the classifier model
        System.out.println("Training regression model...");
        ArrayList<Attribute> features = new ArrayList<Attribute>();
        features.add(new Attribute("impostor"));
        List<String> classLabels = new ArrayList<String>();
        classLabels.add("1");
        classLabels.add("0");

        if (ngrams)
            features.add(new Attribute("ngrams"));
        if (tfidf)
        {
            tfidfVectors = NGramVectors.run(postData, TokenType.TERMS);
            features.add(new Attribute("tfidf"));
        }
        if (writeprints)
        {
            writeprintsVectors = WritePrintsVectors.jstyloVectors(postData, new ArrayList<>(postData.keySet()));
            features.add(new Attribute("writeprints"));
        }
        features.add(new Attribute("class", classLabels));

        Instances trainSet = new Instances("train", features, trainPairs.size());
        trainSet.setClassIndex(trainSet.numAttributes()-1);
        for (int i = 0; i < trainPairs.size(); i++)
        {
            Instance in = similarityVector(trainPairs.get(i).getValue0(), trainPairs.get(i).getValue1(),
                    features, ngrams, ngramVectors, tfidfVectors, writeprintsVectors);
            if (in != null)
            {
                in.setDataset(trainSet);
                in.setClassValue(""+trainLabels[i]);
                trainSet.add(in);
            }
            //			Utilities.printPercentProgress(i, trainPairs.size());
        }
        classifier.batchTrain(trainSet);
//        return null;
        return features;
    }

    protected Instance similarityVector(String s1, String s2, ArrayList<Attribute> features, boolean useNgrams,
                                        Map<String, Vector> ngrams, Map<String, Vector> tfidf, Map<String, Vector> writePrints)
    {
        Instance i = new DenseInstance(features.size());
        int index = 0;
        // always use impostor similarity
        //		ImpostorSimilarity impostorSim = new ImpostorSimilarity(pair.getFirst(), pair.getSecond(), ngrams, bins);
        Vector v1 = ngrams.get(s1);
        Vector v2 = ngrams.get(s2);
        //		i.setValue(features.get(index++), impostorSim.run(impostors)); // impostor similarity

        if (useNgrams)
        {
            i.setValue(features.get(index++), ImpostorUtilities.minMaxSimilarity(v1, v2)); // tfidf minmax
        }

        if (tfidf != null)
        {
            v1 = tfidf.get(s1);
            v2 = tfidf.get(s2);
            i.setValue(features.get(index++), ImpostorUtilities.minMaxSimilarity(v1, v2)); // tfidf minmax
        }
        // writeprints hack
        if (writePrints != null)
        {
            DistanceMeasure cosDistance = new CosineDistanceMeasure();
            v1 = writePrints.get(s1);
            v2 = writePrints.get(s2);

            if (v1 != null && v2 != null)
                i.setValue(features.get(index++), 1 - cosDistance.distance(v1, v2)); // writeprints cosine
            else return null;
        }
        return i;
    }

    protected abstract Map<String, String> setupClusterData(Map<String, Integer> labels, List<String> clusterPosts);

    protected List<String> printResults()
    {
        double precision, recall, accuracy;
        List<String> output = new ArrayList<>();

        System.out.printf("\n%-10s %10s %10s \n", "i", "same author", "diff author");
        for (int i = 0; i < bins; i++)
        {
            System.out.printf("%-10d %10.4f %10.4f\n", i, sameAuthor[i], diffAuthor[i]);
        }

        for (int i = bins-1; i > 0; i--)
        {
            sameAuthor[i-1] += sameAuthor[i];
            diffAuthor[i-1] += diffAuthor[i];
        }
        System.out.printf("\n%-10s %10s %10s %10s\n", "Threshold", "Precision","Recall","Accuracy");

        double maxP = 0;
        double avgP = 0;
        double lastR = 0;
        double points = 0;
        for (int i = 0; i < bins; i++)
        {
            if ((sameAuthor[i]+diffAuthor[i]) > 0)
                precision = sameAuthor[i]/(sameAuthor[i]+diffAuthor[i]);
            else
                precision = 0;
            accuracy = ( (diffAuthor[0] - diffAuthor[i]) + sameAuthor[i]) / (sameAuthor[0] + diffAuthor[0]);
            recall = sameAuthor[i]/sameAuthor[0];
            if (precision < maxP)
                precision = maxP;
            else
                maxP = precision;

            // accumalte avgP
            if (i > 0 && recall != lastR)
            {
                avgP += precision * (lastR-recall);
                points++;
            }
            lastR = recall;

            // threshold , precision, recall, accuracy
            output.add((double)i/bins + "\t" + precision + "\t" + recall + "\t" + accuracy);
            System.out.printf("%-10.4f %10.4f %10.4f %10.4f\n", (double)i/bins, precision, recall, accuracy);
        }
        System.out.println("AVG P = " + avgP/points);
        return output;
    }

    /**
     *
     * @param results [0] disimilaity of pair [1] 1=same pair, 0=diff pair
     */
    public List<String> pairResults(List<Triplet<Double, Double, Integer>> results, String feature)
    {
        List<String> outputLines = new ArrayList<>();

        // sort results from low to high?
        Collections.sort(results);
        double same = 0;
        double different = 0;
        double precision;
        double recall;
        for (Triplet<Double, Double, Integer> p : results)
        {
            if (p.getValue2() == 1)
            {
                same++;
                precision = same / (same+different);
                recall = same / (double)testSize;
                outputLines.add(recall + "\t" + precision + "\t" + feature);
                System.out.println(recall + "\t" + precision + "\t" + feature);
            }
            else
                different++;
        }
        return outputLines;
    }

    public void exportResultsToFile(List<String> outputLines, String outputLocation)
    {
        if (outputLocation != null)
        {
            try(PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(outputLocation, true))))
            {
                for (String line : outputLines)
                {
                    writer.println(line);
                    System.out.println(line);
                }
            }
            catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
    }

    protected void printSimilarityDistribution(String feature)
    {
        String expType = "";
        if (this instanceof BlogExperiment)
            expType = "blog";
        else
            expType = "backpage";
        try(PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter("results/paper/distribution", true))))
        {
            for (int i = 0; i < bins; i++)
            {
                writer.println(i + "\t" + expType + "\t same \t" + sameAuthor[i] + "\t" + feature );
                writer.println(i + "\t" + expType + "\t different \t" + diffAuthor[i] + "\t" + feature );
            }
        }
        catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    /**
     * Updates the same author and different author threshold result counts
     * @param score the impostor similarity
     * @param label 1 if score was from a same author comparison, 0 otherwise
     */
    public static void updateResultCounts(double score, int label, double[] sameCounts, double[] diffCounts)
    {
        int bindex;
        if (score == 0)
            bindex = 0;
        else
            bindex = (int) Math.min(Parameters.BINS-1, (score * Parameters.BINS) );

        if(label > 0)
            sameCounts[bindex]++;
        else
            diffCounts[bindex]++;
    }

    public static void main(String[] args)
    {
        args = new String[]{
                "-t", "cluster",
                "-f", "allpairs",
                "-d", "blog",
                "-ac", "25",
                "-ck", "25",
                "-ipi", "50",
                "-k", "100",
                "-fs", ".4",
//                "-th", ".3",
                "-merge", ".5",
//				"-st",
//                				"-ts", "500",
                "-lsh",
                //				"-k","100",
                //				"-st"
                //				"-c", "regression",
                //				"-city", "georgia",
//                				"-ng", "-tfidf", "-wp",
                "-o", "results/test/",
                "-rf"
//                						"-mi", "10",
//                						"-div", "27"
        };


        Experiment experiment = null;
        ExperimentCmdLineParser cmd = new ExperimentCmdLineParser(args);

        long start = System.currentTimeMillis();

        // choose the dataset to run
        switch(cmd.data)
        {
            case "backpage":
                experiment = new BackPageExperiment(cmd.city);
                break;
            case "blog":
                experiment = new BlogExperiment(cmd.mistakenImpostors, cmd.divisions);
                break;
            case "aaac":
                experiment = new AAACExperiment();
                break;
        }

        assert experiment != null;
        experiment.merge = cmd.merge;
        experiment.testSize = cmd.authors;
        experiment.trainSize = cmd.trainSize;
        experiment.threshold = cmd.threshold;

        // choose the experiment to run
        switch(cmd.type)
        {
            case "cluster":
                chooseClusterMethod(experiment, cmd);
                break;
            case "pairs":
                choosePairMethod(experiment, cmd);
                break;
            case "learn-pairs":
                experiment.learnPairs(cmd.output, cmd.classifier, cmd.ngrams, cmd.tfidf, cmd.writeprints);
                break;
        }
        System.out.println("Took " + (System.currentTimeMillis() - start) +" to complete.");
    }

    private static void choosePairMethod(Experiment experiment, ExperimentCmdLineParser cmd)
    {
        if (cmd.mistakenImpostors > 0 && (experiment.getClass() == BlogExperiment.class))
            experiment.mistakenImpostors(cmd.output, cmd.mistakenImpostors, cmd.divisions);
        else
        {
            TokenType feature = null;
            switch (cmd.feature) {
                case "impostor":
                    feature = TokenType.IMPOSTOR;
                    break;
                case "ngram":
                    feature = TokenType.NGRAMS;
                    break;
                case "tfidf":
                    feature = TokenType.TERMS;
                    break;
                case "stylo":
                    feature = TokenType.STYLO;
                    break;
                case "stems":
                    feature = TokenType.STEMS;
                    break;
            }

            experiment.pairs(cmd.output, feature);
        }
    }


    private static void chooseClusterMethod(Experiment experiment, ExperimentCmdLineParser cmd)
    {
        Map<String, Integer> labels	= new HashMap<>();
        List<String> clusterPosts = new ArrayList<>();
        Map<String, String> data = experiment.setupClusterData(labels, clusterPosts);
        Clusterer cluster = null;
        switch (cmd.feature)
        {
            case "impostor":
                cluster = new ImpostorCluster(data, clusterPosts, experiment.impostors, cmd.useBest);
                break;
            case "ngram":
                cluster = new NGramBaseLineClustering(TokenType.NGRAMS, data, clusterPosts);
                break;
            case "tfidf":
                cluster = new NGramBaseLineClustering(TokenType.TERMS, data, clusterPosts);
                break;
            case "stems":
                cluster = new NGramBaseLineClustering(TokenType.STEMS, data, clusterPosts);
                break;
            case "stylo":
                cluster = new WritePrintsBaseLineClustering(data, clusterPosts);
                break;
            case "compare":
                // impostor
                cluster = new ImpostorCluster(data, clusterPosts, experiment.impostors, cmd.useBest);
                cluster.cluster(labels, cmd.output+"-impostor", "impostor", cmd.lsh, cmd.clusterK);
                // ngram
                cluster = new NGramBaseLineClustering(TokenType.NGRAMS, data, clusterPosts);
                cluster.cluster(labels, cmd.output+"-ngram", "ngram", cmd.lsh, cmd.clusterK);
                // terms
                cluster = new NGramBaseLineClustering(TokenType.TERMS, data, clusterPosts);
                cluster.cluster(labels, cmd.output+"-terms", "terms", cmd.lsh, cmd.clusterK);
                //stems
                cluster = new NGramBaseLineClustering(TokenType.STEMS, data, clusterPosts);
                cluster.cluster(labels, cmd.output+"-stems", "stems", cmd.lsh, cmd.clusterK);
                break;
            // runs experiment comparing all pairs to lsh for impostor and ngram
            case "allpairs":
                // impostor
                cluster = new ImpostorCluster(data, clusterPosts, experiment.impostors, cmd.useBest);
                cluster.cluster(labels, cmd.output+"-impostor-allpairs", "impostor-allpairs", false, cmd.clusterK);
                cluster = new ImpostorCluster(data, clusterPosts, experiment.impostors, cmd.useBest);
                cluster.cluster(labels, cmd.output+"-impostor-lsh", "impostor-lsh", true, cmd.clusterK);
                // ngram
                cluster = new NGramBaseLineClustering(TokenType.NGRAMS, data, clusterPosts);
                cluster.cluster(labels, cmd.output+"-ngram-allpairs", "ngram-allpairs", false, cmd.clusterK);
                cluster = new NGramBaseLineClustering(TokenType.NGRAMS, data, clusterPosts);
                cluster.cluster(labels, cmd.output+"-ngram-lsh", "ngram-lsh", true, cmd.clusterK);
                break;
        }
        if (cluster != null && !cmd.feature.equals("compare") && !cmd.feature.equals("allpairs"))
            cluster.cluster(labels, cmd.output, cmd.feature, cmd.lsh, cmd.clusterK);

    }
}
