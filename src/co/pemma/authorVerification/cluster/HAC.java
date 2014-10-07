package co.pemma.authorVerification.cluster;

import co.pemma.authorVerification.Classifier;
import co.pemma.authorVerification.Parameters;
import co.pemma.authorVerification.Utilities;
import org.apache.commons.math3.util.ArithmeticUtils;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.CosineDistanceMeasure;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import weka.core.DenseInstance;
import weka.core.Instance;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

public class HAC
{
    // map each post key to its index for quick access
    Map<String, Integer> postIndexMap;
    Map<String, Integer> labels;
    String[] indexPostMap;
    String[] clusterPosts;
    List<String>[] clusters;
    Matrix similarityMatrix;
    int clusterCount;

    List<double[]>pairResults;
    List<double[]>perClustResults;
    List<double[]>perAuthResults;


    public HAC(Map<String, Integer> postIndexMap, String[] clusterPosts, Matrix similarityMatrix, Map<String, Integer> labels)
    {
        this.labels = labels;
        this.similarityMatrix = similarityMatrix;
        this.clusterPosts = clusterPosts;
        this.clusters = new List[clusterPosts.length];
        for (int i = 0; i < clusterPosts.length; i++)
        {
            List<String> c = new ArrayList<>();
            c.add(clusterPosts[i]);
            clusters[i] = c;
        }
        this.clusterCount = clusters.length;

        this.postIndexMap = postIndexMap;
        this.indexPostMap = new String[postIndexMap.size()];
        for (Entry<String, Integer> e : postIndexMap.entrySet())
            indexPostMap[e.getValue()] = e.getKey();

        pairResults = new ArrayList<>();
        perClustResults = new ArrayList<>();
        perAuthResults = new ArrayList<>();
    }

    public void cluster(double threshold)
    {
        Pair<Integer, Integer> agglomPair;
        while( (agglomPair = maxSimilarity(threshold)) != null)
        {
            mergeClusters(agglomPair);
        }
        aggregateClusterResults(threshold);
        //		printClusterResultsNormalized(threshold, output);
    }

    public void cluster(int kClusters)
    {
        Pair<Integer, Integer> agglomPair;
        int iteration = 0;
        while( (agglomPair = maxSimilarity(-1)) != null && clusterCount > kClusters)
        {
//            if (clusterCount == kClusters + 1)
//                System.out.println("break;");
            mergeClusters(agglomPair);
            aggregateClusterResults(-1);
            iteration = Utilities.printPercentProgress(iteration++, similarityMatrix.rowSize());
        }
    }

    /**
     * cluster until all items are in a single cluster,
     * print out results after each agglomeration
     */
    public void cluster()
    {
        Pair<Integer, Integer> agglomPair;
        int iteration = 0;
        while( (agglomPair = maxSimilarity(-1)) != null)
        {
            mergeClusters(agglomPair);
            aggregateClusterResults(-1);
            iteration = Utilities.printPercentProgress(iteration++, similarityMatrix.rowSize());
        }
    }


    /**
     * @return find indices of the clusters with max similarity
     */
    public Pair<Integer, Integer> maxSimilarity(double threshold)
    {
        Pair<Integer, Integer> bestPair = null;
        double maxSim = -1.0;
        double val;
        Vector row;

        for (int i = 0; i < similarityMatrix.numRows(); i++)
        {
            row = similarityMatrix.viewRow(i);
            if (row.getNumNonZeroElements() > 0)
            {
                for (Element e : row.nonZeroes())
                {
                    // only care about upper half of matrix
                    if (e.index() > i )
                    {
                        val = e.get();
                        if (val > 0 && val > maxSim)
                        {
                            maxSim = val;
                            bestPair = new Pair<>(i, e.index());
                        }
                    }
                }
            }
        }
        if (maxSim > threshold)
            return bestPair;
        else
            return null;
    }

    /**
     * @return find indices of the clusters with max similarity
     */
    public Pair<Integer, Integer> maxSimilarity(double threshold, Classifier classifier)
    {
        Pair<Integer, Integer> bestPair = null;
        int rowSize, colSize;
        double maxSim = 0, sim;
        Instance in = null;
        Vector row;

        for (int i = 0; i < similarityMatrix.numRows(); i++)
        {
            row = similarityMatrix.viewRow(i);
            if (row.getNumNonZeroElements() > 0)
            {
                rowSize = clusters[i].size();
                for (Element e : row.nonZeroes())
                {
                    // only care about upper half of matrix
                    if (e.index() > i )
                    {
                        colSize = clusters[e.index()].size();
                        in = new DenseInstance(classifier.size);
                        in.setValue(0, e.get());
                        if (classifier.size > 2)
                        {
                            in.setValue(1, Math.min(rowSize, colSize)/Math.max(rowSize, colSize));
                            if (classifier.size > 3)
                                in.setValue(2, rowSize + colSize);
                        }
                        in.setDataset(classifier.getTrainVectors());
                        in.setClassValue(0);
                        sim = classifier.classifyVector(in);
                        if (sim > maxSim)
                        {
                            maxSim = sim;
                            bestPair = new Pair<>(i, e.index());
                        }
                    }
                }
            }
        }
        if (maxSim > threshold)
            return bestPair;
        else
            return null;
    }

    /**
     * Add all of the values in the smaller cluster to the larger one
     * @param agglomeration : indices of the two clusters to merge
     */
    public void mergeClusters(Pair<Integer, Integer> agglomeration)
    {
        int smallIndex = agglomeration.getFirst();
        int largeIndex = agglomeration.getSecond();
        List<String> small = clusters[agglomeration.getFirst()];
        List<String> large = clusters[agglomeration.getSecond()];

        if (small.size() > large.size())
        {
            List<String> temp = large;
            large = small;
            small = temp;
            smallIndex = agglomeration.getSecond();
            largeIndex = agglomeration.getFirst();
        }

        int smallSize = small.size();
        int largeSize = large.size();

        large.addAll(small);

        clusters[smallIndex] = null;
        clusterCount--;
        //		System.out.println();
        //		printMatrix();
        //		System.out.println();

        updateSimilarities(smallIndex, largeIndex, smallSize, largeSize);

        //		System.out.println(smallIndex + "\t" + largeIndex);
        //		System.out.println();
        //		printMatrix();
        //		System.out.println();
    }

    /**
     * average linkage
     * NOTE:  faster to do this manually iterating over non zeros once
     * @param deleteIndex row/column to delete
     * @param keepIndex row/column to update with average of it and delete row/col
     */
    public void updateSimilarities(int deleteIndex, int keepIndex, int deleteSize, int keepSize)
    {
        Vector deleteRow = similarityMatrix.viewRow(deleteIndex);
        Vector deleteCol = similarityMatrix.viewColumn(deleteIndex);
        Vector keepRow = similarityMatrix.viewRow(keepIndex);
        Vector keepCol = similarityMatrix.viewColumn(keepIndex);

        // assignColumn seems fucked, or i dont know what im doing
        //		// average the two rows
        //		Vector avgRow = (keepRow.times(keepSize).plus(deleteRow.times(deleteSize))).divide(keepSize + deleteSize);
        //		similarityMatrix.assignRow(keepIndex, avgRow);
        //		similarityMatrix.assignRow(deleteIndex, zeroVector);
        //
        //		// average the two columns
        //		Vector avgCol = (keepCol.times(keepSize).plus(deleteCol.times(deleteSize))).divide(keepSize + deleteSize);
        //		similarityMatrix.assignColumn(keepIndex, avgRow);
        //		similarityMatrix.assignColumn(deleteIndex, zeroVector);

        // dumb version only takes advantage of sparse matrix in one axis

        for (int i = 0; i < similarityMatrix.columnSize(); i++)
        {
            double dr = deleteRow.get(i);
            double dc = deleteCol.get(i);
            double kr = keepRow.get(i);
            double kc = keepCol.get(i);
            if (dr > 0 || kr > 0)
            {
                double updateVal = ((dr*deleteSize) + (kr*keepSize)) / (deleteSize + keepSize);
                deleteRow.set(i, 0);
                keepRow.set(i, updateVal);

                updateVal = ((dc*deleteSize) + (kc*keepSize)) / (deleteSize + keepSize);
                deleteCol.set(i, 0);
                keepCol.set(i, updateVal);
            }
        }
        similarityMatrix.set(keepIndex, keepIndex, 0);
    }

    /**
     * print the current state of the similarity matrix
     */
    public void printMatrix()
    {
        for (int i = 0; i < similarityMatrix.rowSize(); i++)
        {
            for (int j = 0; j < similarityMatrix.columnSize(); j++)
                System.out.print(similarityMatrix.get(i, j) + "\t");
            System.out.println();
        }
    }

    public void aggregateClusterResults(double threshold)
    {
        // map true label to Pair(index of best match cluster, occurrence of label in cluster)
        Map<Integer, Pair<Integer, Double>> perAuthorBestCluster = new HashMap<>();
        // map cluster index to Pair (best match true label, occurrence of label in cluster)
        Map<Integer, Pair<Integer, Double>> perClusterBestAuthor = new HashMap<>();
        // map true label to its true total count
        Map<Integer, Double> totalClassCounts = new HashMap<>();

        double[] positives = getClusterCounts(perAuthorBestCluster, perClusterBestAuthor, totalClassCounts);
        double tp = positives[0];
        double fp = positives[1];


        // find the best classified output cluster to match each of the authors
        double sameAuthorPairs = authorBestOutputClusterResults(totalClassCounts, perAuthorBestCluster, threshold);
        // find the best author to match each of the classified output clusters
        clusterBestAuthor(totalClassCounts, perClusterBestAuthor, threshold);
        double pairPrecision = 0;
        if ((tp+fp) > 0)
            pairPrecision = tp/(tp+fp);
        double pairRecall =  tp/sameAuthorPairs;
        pairResults.add(new double[]{pairRecall, pairPrecision});
    }

    /**
     * match each output cluster to the author that occurs most in it
     * @param totalClassCounts map each author to its true post total
     * @param perClusterBestAuthor map each cluster to its best true author match
     * @param threshold the threshold clustering stopped at
     */
    private void clusterBestAuthor(Map<Integer, Double> totalClassCounts, Map<Integer, Pair<Integer, Double>> perClusterBestAuthor, double threshold)
    {
        double clusterPrecision = 0, clusterRecall = 0;
        int clustersUsed = 0;
        for (Entry<Integer, Pair<Integer, Double>> e : perClusterBestAuthor.entrySet())
        {
            //			if (clusters[e.getKey()].size() > 1)
            //			{
            Pair<Integer, Double> bestMatch = e.getValue();
            clusterRecall += bestMatch.getSecond() / totalClassCounts.get(bestMatch.getFirst());
            clusterPrecision += bestMatch.getSecond() / clusters[e.getKey()].size();
            clustersUsed++;
            //			}
        }

        if (clustersUsed > 0)
        {
            clusterPrecision = clusterPrecision / clustersUsed;
            clusterRecall = clusterRecall / clustersUsed;
        }
        else
        {
            clusterPrecision = 0;
            clusterRecall = 0;
        }
        perClustResults.add(new double[]{clusterRecall, clusterPrecision});
    }

    /**
     *  match each author to the output cluster it occurs most in
     * @param totalClassCounts map each author to its true post total
     * @param perAuthorBestCluster map each author to the index of its best output cluster
     * @param threshold the threshold clustering stopped at
     * @return total number of same author pairs
     */
    private double authorBestOutputClusterResults(Map<Integer, Double> totalClassCounts, Map<Integer, Pair<Integer, Double>> perAuthorBestCluster, double threshold)
    {
        Pair<Integer, Double> bestMatch;
        double authorPrecision = 0, authorRecall = 0, sameAuthorPairs = 0;
        for (Entry<Integer, Double> e : totalClassCounts.entrySet())
        {
            if (e.getValue().intValue() > 1)
                sameAuthorPairs += ArithmeticUtils.binomialCoefficientDouble(e.getValue().intValue(), 2);
            bestMatch = perAuthorBestCluster.get(e.getKey());
            authorRecall += bestMatch.getSecond() / e.getValue();
            authorPrecision += bestMatch.getSecond() / clusters[bestMatch.getFirst()].size();
        }
        authorPrecision = authorPrecision / totalClassCounts.size();
        authorRecall = authorRecall / totalClassCounts.size();

        perAuthResults.add(new double[]{authorRecall, authorPrecision});

        return sameAuthorPairs;
    }

    public void printResults(String output, String feature)
    {
        System.out.println();
        System.out.println("Recall \t Precision");

//		normalizeResults(pairResults, "pairs");
//		normalizeResults(perAuthResults, "per-author");
//		normalizeResults(perClustResults, "per-cluster");

        if (output != null)
        {
            try (PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(output, true))))
            {
                // pair results - pr[0] = recall, pr[1] = precision
                for (double[] pr : pairResults) {
                    writer.println(pr[0] + "\t" + pr[1] + "\t" + "pairs" + "\t" + feature);
                    System.out.println(pr[0] + "\t" + pr[1] + "\t" + "pairs" + "\t" + feature);
                }
                for (double[] pr : perClustResults) {
                    writer.println(pr[0] + "\t" + pr[1] + "\t" + "per-cluster" + "\t" + feature);
//                    System.out.println(pr[0] + "\t" + pr[1] + "\t" + "per-cluster" + "\t" + feature);
                }
                for (double[] pr : perAuthResults) {
                    writer.println(pr[0] + "\t" + pr[1] + "\t" + "per-author" + "\t" + feature);
//                    System.out.println(pr[0] + "\t" + pr[1] + "\t" + "per-author" + "\t" + feature);
                }
            }
            catch (IOException e1) {
                // TODO Auto-generated catch block
                e1.printStackTrace();
            }
        }
    }

    public double silhouette(Map<String, Vector> vectorMap)
    {
        double avgSi = 0;
        double totalPoints = 0;
        double ai, bi, si;
        List<String> cluster1, cluster2;

        for(int i = 0; i < clusters.length; i++)
        {
            cluster1 = clusters[i];
            if (cluster1 != null) {
                for (String post : cluster1) {
                    ai = 0;
                    bi = 1;
                    if (cluster1.size() > 0) {
                        for (int j = 0; j < clusters.length; j++) {
                            cluster2 = clusters[j];
                            if (cluster2 != null) {
                                // find intra cluster dissimilarity for this post
                                if (i == j)
                                    ai = avgDissimilarityToCluster(post, cluster2, vectorMap);
                                    // find the best matching neighbor cluster for this post
                                else {
                                    double avgDis = avgDissimilarityToCluster(post, cluster2, vectorMap);
                                    if (avgDis < bi)
                                        bi = avgDis;
                                }
                            }
                        }
                    }
                    if (ai < bi)
                        si = 1 - (ai / bi);
                    else if (ai > bi)
                        si = (bi / ai) - 1;
                    else
                        si = 0;
                    avgSi += si;
                    totalPoints++;
                }
            }
        }
        if (totalPoints > 0 )
            return avgSi/totalPoints;
        else {
            System.err.println("no points used in silhouete calculation for some reason");
            return 0;
        }
    }

    public double avgDissimilarityToCluster(String post, List<String> cluster, Map<String,Vector> vectorMap)
    {
        CosineDistanceMeasure cosine = new CosineDistanceMeasure();
        double dissimilarity = 0;
        double posts = cluster.size();
        int index1 = postIndexMap.get(post);
        for (String post2 : cluster)
        {
            int index2 = postIndexMap.get(post2);
            // intracluster dissimilarity, -1 for self from the total
            if (index1 == index2)
                posts--;
            else
            {
//                double d = (1-similarityMatrix.get(index1, index2));
                double d = cosine.distance(vectorMap.get(indexPostMap[index1]), vectorMap.get(indexPostMap[index2]));
                dissimilarity += d;
            }
        }
        if (posts == 0)
            return 0;
        else
            return dissimilarity/posts;
    }


    /**
     * remove 0 rows caused by converting the threshold to index
     * make 'precision(r) = max(p_i) where (r_i,p_i) are the calculated points and r<=r_i'
     * --- as recall drops, precision cant (this is what IR people do they say)
     */
    private void normalizeResults(double[][] result, String metric)
    {
        double maxP = 0;
        for (int i = 0; i < Parameters.BINS; i++){
            // because threshold is double cast to int, we might have 0 buckets
            if (i > 0 && result[i][0] == 0 && result[i][1] == 0)
            {
                result[i][0] = result[i-1][0];
                result[i][1] = result[i-1][1];
            }
            else
            {
                // as recall drops, precision cant drop
                if (i > 0 && result[i][0] < maxP)
                    result[i][0] = maxP;
                else
                    maxP = result[i][0];
            }
            System.err.println(i + "\t" + result[i][0] + "\t" +result[i][1] + "\t" + metric);
        }
    }

    /**
     *
     * @param perAuthorBestCluster fill with map from each author to its best matching output cluster
     * @param perClusterBestAuthor fille with each output cluster mapped to its best author
     * @param totalClassCounts fill with map from each author to its count total
     * @return [0] true positive pairings [1] false positive pairings in output clusters
     */
    private double[] getClusterCounts(
            Map<Integer, Pair<Integer, Double>> perAuthorBestCluster,
            Map<Integer, Pair<Integer, Double>> perClusterBestAuthor,
            Map<Integer, Double> totalClassCounts) {
        Map<Integer, Double> clusterCounts;
        int label, label2;
        double[] positives = new double[] {0, 0};
        double count;
        String point1, point2;
        for(int i = 0; i < clusters.length; i++)
        {
            List<String> cluster = clusters[i];
            if (cluster != null)
            {
                clusterCounts = new HashMap<>();
                //								System.out.println();
                for (int j = 0; j < cluster.size(); j++)
                {
                    point1 = cluster.get(j);
                    label = labels.get(point1);
                    //					System.out.print(label + "\t");

                    // keep track of label counts within this cluster
                    if (clusterCounts.containsKey(label))
                        clusterCounts.put(label, clusterCounts.get(label) + 1);
                    else
                        clusterCounts.put(label, 1.);

                    for (int k = j+1; k <cluster.size(); k++)
                    {
                        point2 = cluster.get(k);
                        label2 = labels.get(point2);
                        if (label == label2)
                            positives[0]++;
                        else
                            positives[1]++;
                    }
                }

                double maxCount = 0;
                int maxLabel = -1;
                // update the total counts and best match counts
                for (Entry<Integer, Double> e : clusterCounts.entrySet())
                {
                    label = e.getKey();
                    count = e.getValue();
                    // update the total counts for the true label
                    if (totalClassCounts.containsKey(label))
                        totalClassCounts.put(label, totalClassCounts.get(label) + count);
                    else
                        totalClassCounts.put(label, count);

                    // keep track of the most occurring label in this cluster
                    if (count > maxCount)
                    {
                        maxCount = count;
                        maxLabel = label;
                    }
                    // see if this is the best match for this label
                    if (perAuthorBestCluster.containsKey(label))
                    {
                        if (count > perAuthorBestCluster.get(label).getSecond())
                            perAuthorBestCluster.put(label, new Pair<>(i, count));
                    }
                    else
                        perAuthorBestCluster.put(label, new Pair<>(i, count));
                }
                perClusterBestAuthor.put(i, new Pair<>(maxLabel, maxCount));
            }
        }
        return positives;
    }


    //	public static void main(String[] args)
    //	{
    //		test();
    //	}
    //
    //	private static void test()
    //	{
    //		SparseMatrix mat = new SparseMatrix(4, 4);
    //		mat.set(0, 1, .1);
    //		mat.set(0, 2, .2);
    //		mat.set(0, 3, .0);
    //
    //		mat.set(1, 0, .1);
    //		mat.set(1, 2, .3);
    //		mat.set(1, 3, .5);
    //
    //		mat.set(2, 0, .2);
    //		mat.set(2, 1, .3);
    //		mat.set(2, 3, .4);
    //
    //		mat.set(3, 0, .0);
    //		mat.set(3, 1, .5);
    //		mat.set(3, 2, .4);
    //
    //
    //		String[] c = new String[]{"a","b","c", "d"};
    //
    //		Map<String, Integer> pim = new HashMap<>();
    //		pim.put("a", 0);
    //		pim.put("b", 1);
    //		pim.put("c", 2);
    //		pim.put("d", 3);
    //
    //		HAC hac = new HAC(pim, c, mat, null);
    //		hac.cluster(.3);
    //	}
}