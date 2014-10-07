package co.pemma.authorVerification.cluster;

import co.pemma.authorVerification.ImpostorUtilities;
import co.pemma.authorVerification.Parameters;
import com.google.common.util.concurrent.AtomicDouble;
import org.apache.mahout.common.Pair;
import org.apache.mahout.math.SparseMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;

import java.io.*;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

public abstract class Clusterer
{
    SparseMatrix similarityMatrix;
    String[] clusterPosts;
    String[] indexPostMap;

    Map<String, Vector> vectorMap;
    // map each post key to its index for quick access
    Map<String, Integer> postIndexMap;
    Map<String, String> posts;
    int featureCount;

    AtomicDouble[] sameCounts;
    AtomicDouble[] diffCounts;
    AtomicInteger pairsTried;

    boolean randomSamplePairs = false;


    public Clusterer(Map<String, String> posts, List<String> clusterPostList)
    {
        this.clusterPosts = clusterPostList.toArray(new String[clusterPostList.size()]);
        this.posts = posts;
        // initialize result arrays
        sameCounts = new AtomicDouble[Parameters.BINS];
        diffCounts = new AtomicDouble[Parameters.BINS];
        for (int i = 0; i < Parameters.BINS; i++)
        {
            sameCounts[i] = new AtomicDouble(0);
            diffCounts[i] = new AtomicDouble(0);
        }
    }

    protected void createIndexMaps()
    {
        this.postIndexMap = new ConcurrentHashMap<>();
        // map each post to its index in the array
        for (int i = 0; i < clusterPosts.length; i++)
            postIndexMap.put(clusterPosts[i], i);

        this.indexPostMap = new String[postIndexMap.size()];
        for (Entry<String, Integer> e : postIndexMap.entrySet())
            indexPostMap[e.getValue()] = e.getKey();
    }

    protected void nearestNeighbor(Map<String, Integer> labels)
    {
        Vector row;
        double maxSim;
        boolean same;
        int bestMatch = -1, xlab, ylab;
        for (int i = 0; i < similarityMatrix.rowSize(); i++)
        {
            row = similarityMatrix.viewRow(i);
            maxSim = 0;
            if (row.getNumNonZeroElements() > 0)
            {
                for (Element e : row.nonZeroes())
                {
                    if (e.get() > maxSim)
                    {
                        maxSim = e.get();
                        bestMatch = e.index();
                    }
                }
                xlab = labels.get(indexPostMap[i]);
                ylab = labels.get(indexPostMap[bestMatch]);
                if (xlab == ylab)
                    same = true;
                else {
                    same = false;
                    // print mistake
                    System.out.println(indexPostMap[i] + ":" + indexPostMap[bestMatch]);
                }

                updateResultCounts(maxSim, same, sameCounts, diffCounts);
            }
        }
//        printResults(null);
    }

    private void printResults(String output)
    {
        for (int i = Parameters.BINS-1; i > 0; i--)
        {
            sameCounts[i-1].getAndAdd(sameCounts[i].get());
            diffCounts[i-1].getAndAdd(diffCounts[i].get());
        }

        System.out.println("Threshold \t Same \t Diff");
        for (int i = 0; i < Parameters.BINS; i++)
        {
            System.out.println(i + "\t" + sameCounts[i].get() + "\t"+ +diffCounts[i].get());
            if (output != null)
            {
                try(PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(output, true))))
                {
                    writer.println(i + "\t" + sameCounts[i].get() + "\t"+ +diffCounts[i].get() + "\t"
                            + Parameters.paramsToString());
                }
                catch (IOException e) {
                    // TODO Auto-generated catch block
                    e.printStackTrace();
                }
            }
        }
        System.out.println("COUNTS: " +sameCounts[0] + "\t" +diffCounts[0]);
    }

    protected void printRandomePairSample(List<Pair<String, String>> pairs, Map<String, Integer> labels)
    {
        Pair<String, String> p;
        String p1, p2;
        int samples = 100, used = 0;
        try(PrintWriter writer = new PrintWriter("random-pair-samples"))
        {
            for (int i : ImpostorUtilities.randomIndices(pairs.size(), pairs.size()))
            {
                p = pairs.get(i);
                p1 = p.getFirst();
                p2 = p.getSecond();

                if (!labels.get(p1).equals(p2))
                {
                    writer.println(posts.get(p1));
                    writer.println();
                    writer.println(posts.get(p2));
                    writer.println("\n\n\n");
                    if (used++ == samples)
                        break;
                }
            }
        }
        catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    /**
     * Uses thread-safe doubles to update the same author and different author threshold result counts
     * @param score the impostor similarity
     * @param same true if score was from a same author comparison, false otherwise
     */
    public static void updateResultCounts(double score, boolean same, AtomicDouble[] sameCounts, AtomicDouble[] diffCounts)
    {
        int bindex;
        if (score == 0)
            bindex = 0;
        else
            bindex = (int) Math.min(Parameters.BINS-1, (score * Parameters.BINS) );

        if(same)
            sameCounts[bindex].getAndAdd(1);
        else
            diffCounts[bindex].getAndAdd(1);
    }

    public abstract void cluster(Map<String, Integer> labels, String output, String feature, boolean lsh, int k);


    public void cluster(Map<String, Integer> labels, String output, String feature)
    {
        HAC hac = new HAC(postIndexMap, clusterPosts, similarityMatrix, labels);
        System.out.println("Using constructed similarity matrix to cluster...");
	    hac.cluster();
        hac.printResults(output, feature);
    }

    public void cluster(Map<String, Integer> labels, String output, String feature, int k)
    {
        HAC hac = new HAC(postIndexMap, clusterPosts, similarityMatrix, labels);
        System.out.println("Using constructed similarity matrix to cluster...");

        hac.cluster(k);
        hac.printResults(output, feature);
        System.out.println(hac.silhouette(vectorMap));
//        sampleClusters(hac.clusters, labels, feature+"-"+k);
    }

    /**
     * Print the output clusters to disk
     * @param clusters output clusters
     * @param labels truth labels
     * @param output output directory
     */
    public void sampleClusters(List<String>[] clusters, Map<String, Integer> labels, String output)
    {
        String baseOutput = "results/sample-clusters/";
        String outputDir = baseOutput + output;
        new File(outputDir).mkdir();
        Random rand = new Random();

        for (List<String> cluster : clusters)
        {
            if (cluster != null && cluster.size() >= 2) {
                try (PrintWriter writer = new PrintWriter(outputDir + "/" + cluster.size()+"-"+rand.nextInt()))
                {
                    for (String postId : cluster) {
                        String post = posts.get(postId);
                        writer.println(postId + ":" + labels.get(postId) + "\n");
                        writer.println(post + "\n");
                        writer.println("\n\n\n");
                    }
                }
                catch (FileNotFoundException e) {
                    e.printStackTrace();
                }
            }
        }
    }
}