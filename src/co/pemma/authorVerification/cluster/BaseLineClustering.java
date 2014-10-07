package co.pemma.authorVerification.cluster;

import co.pemma.authorVerification.ImpostorUtilities;
import co.pemma.authorVerification.Utilities;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.CosineDistanceMeasure;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.SparseMatrix;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class BaseLineClustering extends Clusterer
{
    boolean MIN_MAX;

    public BaseLineClustering(Map<String, String> posts, List<String> clusterPosts, boolean useMinMax)
    {
        super(posts, clusterPosts);
        similarityMatrix = new SparseMatrix(this.clusterPosts.length, this.clusterPosts.length);
        this.MIN_MAX = useMinMax;
        createIndexMaps();
    }

    @Override
    public void cluster(Map<String, Integer> labels, String output, String feature, boolean lsh, int k)
    {
        createSimilarityMatrix(labels, output, lsh);
        nearestNeighbor(labels);
        cluster(labels, output, feature, k);
    }

    private SparseMatrix createSimilarityMatrix(
            Map<String, Integer> labels, String output, boolean lsh)
    {
        System.out.println("Running all pairs comparisons...");

        Partitioner partitioner = new Partitioner(clusterPosts, vectorMap);
        List<Pair<String, String>> pairs;

        if (lsh)
        {
            pairs = partitioner.lshPartitions(labels);
            if (randomSamplePairs)
                printRandomePairSample(pairs, labels);
        }
        else
        {
            pairs = new ArrayList<>();
            for (int i = 0; i < clusterPosts.length; i++)
                for (int j = i+1; j < clusterPosts.length; j++)
                    pairs.add(new Pair<>(clusterPosts[i], clusterPosts[j]));
        }

        int start, end, threads = 1;//Utilities.MAX_THREADS;
        int division = pairs.size()/threads;
        ExecutorService executor = Executors.newFixedThreadPool(threads);

        for (int i = 0; i < threads; i++)
        {
            start = i*division;
            if ( i+1 < threads)
                end = start + division;
            else
                end = pairs.size();
            executor.submit(new BaseLineSimilarityThread(start, end, pairs));
        }
        Utilities.waitForThreads(executor);

        return similarityMatrix;
    }

    public class BaseLineSimilarityThread implements Runnable
    {
        private List<Pair<String, String>> pairs;
        private int start, end;
        private DistanceMeasure distance;

        BaseLineSimilarityThread(int start, int end, List<Pair<String, String>> pairs)
        {
            this.distance = new CosineDistanceMeasure();
            this.pairs = pairs;
            this.start = start;
            this.end = end;
        }

        @Override
        public void run()
        {
            double similarity = 0;
            String x, y ;

            for ( int i = start; i < end; i++)
            {
                x = pairs.get(i).getFirst();
                y = pairs.get(i).getSecond();

                if (MIN_MAX)
                    similarity = ImpostorUtilities.minMaxSimilarity(vectorMap.get(x), vectorMap.get(y));
                else
                    similarity = 1 - distance.distance(vectorMap.get(x), vectorMap.get(y));

                similarityMatrix.set(postIndexMap.get(x), postIndexMap.get(y), similarity);
                similarityMatrix.set(postIndexMap.get(y), postIndexMap.get(x), similarity);

            }
            Utilities.printPercentProgress(pairsTried.incrementAndGet(), clusterPosts.length);
        }
    }
}