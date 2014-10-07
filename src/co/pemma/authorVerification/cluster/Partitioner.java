package co.pemma.authorVerification.cluster;

import co.pemma.authorVerification.ImpostorUtilities;
import co.pemma.authorVerification.Utilities;
import org.apache.mahout.clustering.streaming.cluster.StreamingKMeans;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.CosineDistanceMeasure;
import org.apache.mahout.math.*;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.neighborhood.LocalitySensitiveHashSearch;
import org.apache.mahout.math.neighborhood.UpdatableSearcher;
import org.apache.mahout.math.random.WeightedThing;

import java.io.*;
import java.util.*;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;

public class Partitioner 
{
	int HASH_SEARCH_SIZE = 25;
	boolean randSample = true;

	String[] clusterPosts;
	Map<String, Vector> vectorMap;

	public Partitioner(	String[] clusterPosts, Map<String, Vector> vectorMap)
	{
		this.clusterPosts = clusterPosts; 
		this.vectorMap = vectorMap;
	}

	public List<Pair<String, String>> kMeansPartitions(Map<String, Integer> labels, int topKPartitions)
	{
		// run several rough clusterings and combine results, maybe better?
		List<StreamingKMeans> centroidList = new ArrayList<>();
		for (UpdatableSearcher searcher : new ArrayList<UpdatableSearcher>(){{
			//					add(new FastProjectionSearch (new CosineDistanceMeasure(), 10, 10));
			add(new LocalitySensitiveHashSearch(new CosineDistanceMeasure(), clusterPosts.length));
			//					add(new FastProjectionSearch (new EuclideanDistanceMeasure(), 10, 10));
			//					add(new LocalitySensitiveHashSearch(new EuclideanDistanceMeasure(), 10));
		}})
		{
			Matrix m = new SparseMatrix(vectorMap.size(), vectorMap.get(clusterPosts[0]).size());
			for (int i : ImpostorUtilities.randomIndices(clusterPosts.length, clusterPosts.length))
				m.assignRow(i, vectorMap.get(clusterPosts[i]));

			System.out.println("Running rough partition... ");
			StreamingKMeans partitionCentroids = new StreamingKMeans(searcher, 1);
			searcher = partitionCentroids.cluster(m);
			centroidList.add(partitionCentroids);
		}


		return (partitionData(centroidList, labels, topKPartitions));
	}

	private List<Pair<String, String>> partitionData(List<StreamingKMeans> centroidList, Map<String, Integer> labels, int topKPartitions) 
	{
		long start = System.currentTimeMillis();
		System.out.print("Assigning data to top " + topKPartitions + " partitions");
		// see how the partitioning went
		Map<Integer, List<String>> partitions = new ConcurrentHashMap<>();
		// iterate over each post to be clustered
		for(String key: clusterPosts)
		{
			// in case of overlapping centroid keys between kmean iterations
			int keyModifer = 0;
			for (StreamingKMeans centroids : centroidList)
				keyModifer = findTopPartitions(centroids, partitions, topKPartitions, key, keyModifer);
		}

		Set<String> seenPairs = new HashSet<>();
		String pairString;
		int sameAuthorPairs = 0;
		int totalPairs = 0;

		List<Pair<String, String>> pairs = new ArrayList<>();
		Pair<String, String> pair;
		// iterate over each partition
		for (List<String> lists : partitions.values())
		{
			// count the number of same author pairs in each partition
			for (int i = 0; i < lists.size(); i++)
			{
				int lab1 = labels.get(lists.get(i));
				for (int j = i+1; j < lists.size(); j++)
				{
					if (lists.get(i).compareTo(lists.get(j)) > 0)
					{
						pairString = lists.get(i) + "-" + lists.get(j);
						pair = new Pair<>(lists.get(i), lists.get(j));
					}
					else
					{
						pairString = lists.get(j) + "-" + lists.get(i);
						pair = new Pair<>(lists.get(j), lists.get(i));
					}

					if (!seenPairs.contains(pairString))
					{
						seenPairs.add(pairString);
						pairs.add(pair);
						totalPairs++;
						int lab2 = labels.get(lists.get(j));
						if (lab1 == lab2)
							sameAuthorPairs++;
					}
				}
			}
		}
		System.out.println("  took : " + (System.currentTimeMillis() - start));

		System.out.println("Same author pairs in partitions: " + sameAuthorPairs);
		System.out.println("Total pairs in partitions: " + totalPairs);
		System.out.println("done");
		printResult(topKPartitions, sameAuthorPairs, totalPairs);

		return pairs;
	}

	private void printResult(int topKPartitions, int sameAuthorPairs, int totalPairs) {
        if (new File("results/hac/partitions").exists())
		try(PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter("results/hac/partitions", true))))
		{
			writer.println(clusterPosts.length + "\t" + topKPartitions + "\t" + sameAuthorPairs + "\t" + totalPairs);	
		} 
		catch (IOException e) {
			// TODO Auto-generated catch block	
			e.printStackTrace();
		}
	}

	private int findTopPartitions(StreamingKMeans centroids, Map<Integer, List<String>> partitions, int topKPartitions, String key, int keyModifer) 
	{
		List<String> list;
		Vector v1 = vectorMap.get(key);

		int[] matches = new int[topKPartitions];
		int maxKey = 0;
		for (int i = 0; i < matches.length; i++)
			matches[i] = -1;
		double mostSim = 0, sim;

		// compare to each centroid
		for (Centroid v2 : centroids)
		{
			sim = ImpostorUtilities.minMaxSimilarity(v1, v2.getVector());
			if (sim > mostSim)
			{
				mostSim = sim;
				for (int i = matches.length-1; i > 0; i--)
					matches [i] = matches[i-1];
				matches[0] = v2.getKey() + keyModifer;
				// keep track of highest key for future iterations of method
				if (matches[0] > maxKey)
					maxKey = matches[0];
			}
		}
		// add to top matches list
		for (int match : matches)
		{
			if (partitions.containsKey(match))
				list = partitions.get(match);
			else
			{
				list = new ArrayList<>();
			}
			if (!list.contains(key))
				list.add(key);
			partitions.put(match, list);
		}
		return maxKey + 1;
	}

	public List<Pair<String, String>> lshPartitions(Map<String, Integer> labels, int hashSize)
	{
		this.HASH_SEARCH_SIZE = hashSize;
		return lshPartitions(labels);
	}

	public List<Pair<String, String>> lshPartitions(Map<String, Integer> labels)
	{
		System.out.println("Generating LSH partitions...");
		UpdatableSearcher lsh = new LocalitySensitiveHashSearch(new CosineDistanceMeasure(), clusterPosts.length/5);
		for(int i = 0; i < clusterPosts.length; i++)
		{
			String key = clusterPosts[i];
			lsh.add(new NamedVector(vectorMap.get(key),key));
			Utilities.printPercentProgress(i, clusterPosts.length);
		}
		List<Pair<String, String>> pairs = getPairs(labels, lsh);
		
//		pairs = insertLabelErrors(labels, pairs, 0);
		
		return pairs;
	}

	private List<Pair<String, String>> insertLabelErrors(Map<String, Integer> labels, List<Pair<String, String>> pairs, double unlabelPercent) 
	{
		// identify same author Pairs
		String p1, p2;
		Map<String, Integer> sameAuthorOccurance = new HashMap<>();
		for (Pair<String, String> p : pairs)
		{
			p1 = p.getFirst();
			p2 = p.getSecond();

			if (!labels.get(p1).equals(p2))
			{
				if (sameAuthorOccurance.containsKey(p1))
					sameAuthorOccurance.put(p1, sameAuthorOccurance.get(p1)+1);
				else
					sameAuthorOccurance.put(p1, 1);
				if (sameAuthorOccurance.containsKey(p2))
					sameAuthorOccurance.put(p2, sameAuthorOccurance.get(p2)+1);
				else
					sameAuthorOccurance.put(p2, 1);
			}
		}
		
		Entry<String, Integer> e;
		int unlabeled = 0, index, newLabel;
		Random rand = new Random();
		List<Entry<String, Integer>> posts = new ArrayList<>(sameAuthorOccurance.entrySet());
		while (unlabeled < pairs.size()*unlabelPercent)
		{
			index = rand.nextInt(posts.size());
			e = posts.get(index);
			// change this posts label
			newLabel = labels.get( e.getKey());
			labels.put(e.getKey(), (newLabel+rand.nextInt(pairs.size()))*-1 );
			unlabeled += e.getValue();
			posts.remove(index);
		}
		
		// randomly mis
		return pairs;
	}

	private List<Pair<String, String>> getPairs(Map<String, Integer> labels, UpdatableSearcher searcher) 
	{
		System.out.println("Querying LSH for KNN of each post...");
		List<Pair<String, String>> pairs = new ArrayList<>();
		Pair<String, String> pair;
		String pairString;

		int totalPairs = 0, samePairs = 0;
		Set<String> seenPairs = new HashSet<>();

		int i = 0;
		for(String key : clusterPosts)
		{
			int xLabel = labels.get(key);
			for(WeightedThing<Vector> v : searcher.search(vectorMap.get(key), HASH_SEARCH_SIZE))
			{
				NamedVector nv = (NamedVector) v.getValue();
				if (!nv.getName().equals(key))
				{
					int yLabel = labels.get(nv.getName());
					if (key.compareTo(nv.getName()) > 0)
					{
						pairString = key + "-" + nv.getName();
						pair = new Pair<>(key, nv.getName());
					}
					else
					{
						pairString = nv.getName() + "-" + key;
						pair = new Pair<>(nv.getName(), key);
					}
					if (!seenPairs.contains(pairString))
					{
						totalPairs++;
						seenPairs.add(pairString);
						pairs.add(pair);

						if (xLabel == yLabel)
						{
							samePairs++;
						}
					}
				}
			}	
			i = Utilities.printPercentProgress(i, clusterPosts.length);
		}
		System.out.println("total pairs in LSH: " + totalPairs);
		System.out.println("same author pairs in LSH: " + samePairs);
		printResult(HASH_SEARCH_SIZE, samePairs, totalPairs);
		
		return pairs;
	}

}
