package co.pemma.authorVerification.cluster;

import co.pemma.authorVerification.ImpostorSimilarity;
import co.pemma.authorVerification.ImpostorUtilities;
import co.pemma.authorVerification.NGramVectors;
import co.pemma.authorVerification.Parameters;
import co.pemma.authorVerification.Utilities;
import org.apache.mahout.common.Pair;
import org.apache.mahout.math.SparseMatrix;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

public class ImpostorCluster extends Clusterer
{
    List<String> impostors;
    Map<String, List<String>> bestImpostersMap;
    boolean useBest;

    public ImpostorCluster(Map<String, String> posts, List<String> clusterPosts, List<String> impostors, boolean useBest)
    {
        super (posts, clusterPosts);
        this.postIndexMap			= new HashMap<>();
        this.similarityMatrix 		= new SparseMatrix(clusterPosts.size(), clusterPosts.size());
        this.impostors  			= impostors;

        // generate the vectors for the problem set and imposters
        this.vectorMap = NGramVectors.run(posts);
        this.featureCount = vectorMap.get(impostors.get(0)).size();

        // calculate best imposters for each post to save time in future
        String[] clusterPostArray = clusterPosts.toArray(new String[clusterPosts.size()]);
        this.useBest = useBest;
        if (useBest)
            this.bestImpostersMap = ImpostorUtilities.findTopImposters(impostors, vectorMap, clusterPostArray);

        createIndexMaps();

    }

    @Override
    public void cluster(Map<String, Integer> labels, String output, String feature, boolean lsh, int k)
    {
        createSimilarityMatrix(labels, lsh);
		nearestNeighbor(labels);
        cluster(labels, output, feature, k);
    }

    public SparseMatrix createSimilarityMatrix(Map<String, Integer> labels, boolean lsh)
    {
        boolean kmeans = false;

        Partitioner partitioner = new Partitioner(clusterPosts, vectorMap);

        pairsTried = new AtomicInteger(0);
//        for (int i = 0; i < Parameters.K; i++)
//        {
//            sameCounts[i] = new AtomicDouble(0);
//            diffCounts[i] = new AtomicDouble(0);
//        }

        List<Pair<String, String>> pairs;

        if (kmeans)
            pairs = partitioner.kMeansPartitions(labels, 2);

        else if (lsh)
        {
            //			for (int hashSize : new int[]{25, 50, 100, 250, 500})
            pairs = partitioner.lshPartitions(labels);//, hashSize);
        }
        else
        {
            pairs = new ArrayList<>();
            for (int i = 0; i < clusterPosts.length; i++)
                for (int j = i+1; j < clusterPosts.length; j++)
                    pairs.add(new Pair<>(clusterPosts[i], clusterPosts[j]));
        }
        runPairComparisons(pairs);

        return similarityMatrix;
    }

    private void runPairComparisons(List<Pair<String, String>> pairs)
    {
        System.out.println("Running all pairs comparisons...");

        List<boolean[]> randFeatures = new ArrayList<>();
        List<List<Integer>> randImposters = new ArrayList<>();
        for (int k = 0; k < Parameters.K; k ++)
        {
            randFeatures.add(ImpostorUtilities.randomFeatures(featureCount, Parameters.FEATURE_SUBSET_PERCENT));
            randImposters.add(ImpostorUtilities.randomIndices(Math.min(Parameters.IMPOSTER_COUNT, impostors.size()), Math.min(impostors.size(), Parameters.IMPOSTERS_PER_IT)));
        }

        int start, end;
        int threads = Utilities.MAX_THREADS;
        int division = pairs.size()/threads;
        ExecutorService executor = Executors.newFixedThreadPool(threads);

        for (int i = 0; i < threads; i++)
        {
            start = i*division;
            if ( i+1 < threads)
                end = start + division;
            else
                end = pairs.size();
            executor.submit(new CompareThread(start, end, randFeatures, randImposters, pairs));
        }
        Utilities.waitForThreads(executor);

        //		printResults(output, topKPartitions);
        System.out.println("PAIRS : " + pairsTried);
    }

    public class CompareThread implements Runnable
    {
        List<List<Integer>> randImposters;
        List<boolean[]> randFeatures;
        List<Pair<String, String>> pairs;
        private int start, end;


        CompareThread(int start, int end, List<boolean[]> randFeatures, List<List<Integer>> randImposters, List<Pair<String, String>> pairs)
        {
            this.randImposters = randImposters;
            this.randFeatures = randFeatures;
            this.pairs = pairs;
            this.start = start;
            this.end = end;
        }

        @Override
        public void run()
        {
            double similarity;
            String x, y ;

            for ( int i = start; i < end; i++)
            {
                x = pairs.get(i).getFirst();
                y = pairs.get(i).getSecond();

                similarity = 0;

                Vector xVector = vectorMap.get(x);
                Vector yVector = vectorMap.get(y);

                List<String> xImps;
                List<String> yImps;
                if (useBest) {
                    xImps = bestImpostersMap.get(x);
                    yImps = bestImpostersMap.get(y);
                }
                else
                {
                    xImps = impostors;
                    yImps = impostors;
                }

                for (int k = 0; k < Parameters.K; k++)
                {
                    List<String> randXImps = new ArrayList<>(Parameters.IMPOSTERS_PER_IT);
                    List<String> randYImps = new ArrayList<>(Parameters.IMPOSTERS_PER_IT);

                    for (int dex : randImposters.get(k))
                    {
                        randXImps.add(xImps.get(dex));
                        randYImps.add(yImps.get(dex));
                    }
                    similarity += ImpostorSimilarity.similarityIteration(xVector, yVector, randXImps, randYImps, randFeatures.get(k), vectorMap, x, y);
                }
//				System.out.println(similarity + "\t" + similarity/(double)(Parameters.K*2.));
                double impWeight = .78;
                double minmaxWeight = 1.0-impWeight;
                similarity = (impWeight*(similarity / (Parameters.K*2.))) +
                        (minmaxWeight*(ImpostorUtilities.minMaxSimilarity(xVector, yVector)));

                similarityMatrix.set(postIndexMap.get(x), postIndexMap.get(y), similarity);
                similarityMatrix.set(postIndexMap.get(y), postIndexMap.get(x), similarity);

                Utilities.printPercentProgress(pairsTried.incrementAndGet(), pairs.size());
            }
        }
    }
}
