package co.pemma.authorVerification;

import com.google.common.primitives.Ints;
import org.apache.commons.math3.random.RandomDataGenerator;
import org.apache.mahout.common.Pair;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class ImpostorUtilities 
{
	private static int[] randomFeatureList;
	private static Random rand = new Random();
    private static RandomDataGenerator randomDataGenerator= new RandomDataGenerator();

	public static Vector getSubVector(Vector vector, List<Integer> featureSet)
	{
		Vector subset = new RandomAccessSparseVector(featureSet.size());
		for (int i = 0; i < featureSet.size(); i++)
			subset.set(i, vector.get(featureSet.get(i)));
		return subset;
	}

	/**
	 * select a random subset of feature indices for impostor similarity
	 * @param totalSize size of vectors to consider
	 * @param subsetFraction percentage of features to randomly select
	 * @return true at the selected feature indices, false otherwise
	 */
	public static boolean[] randomFeatures(int totalSize, double subsetFraction)
	{
		if (randomFeatureList == null || randomFeatureList.length != totalSize)
		{
			randomFeatureList = new int[totalSize];
			for (int i = 0; i < totalSize; i++)
				randomFeatureList[i] = i;
		}

		// select subsetFraction of the randomly selected features
		boolean[] indices = new boolean[totalSize];
		int j, temp;
		// fischer yates
		for (int i = totalSize-1; i > totalSize*subsetFraction; i--)
		{
			j = rand.nextInt(i);
			temp = randomFeatureList[i];
			randomFeatureList[i] = randomFeatureList[j];
			randomFeatureList[j] = temp;
			indices[j] = true;
		}	
		return indices;		
	}

	/**
	 * Return a random subset of the values in the range 0 and (totalSize-1), inclusive.
	 * @param totalSize how big is the array
	 * @param subsetSize how many random indices do you want
	 * @return list of random indicies of size subset size
	 */
	public static List<Integer> randomIndices(int totalSize, int subsetSize)
	{
        int[] randomIndexArray = randomDataGenerator.nextPermutation(totalSize, subsetSize);
        return Ints.asList(randomIndexArray);
    }

	/**
	 * calculate the min max distance between two vectors
	 * @param v1
	 * @param v2
	 * @return min-max distance between v1 and v2
	 */
	public static double minMaxSimilarity(Vector v1, Vector v2) 
	{
		double min = 0;
		double max = 0;		

		Iterator<Element> i1 = v1.nonZeroes().iterator();
		Iterator<Element> i2 = v2.nonZeroes().iterator();

		Element e1 = null, e2 = null;

		// if a vector is 0, result will always be 0 
		if (!i1.hasNext() || !i2.hasNext())
			return 0;

		e2 = i2.next();
		while (i1.hasNext())
		{	
			e1 = i1.next();				
			// all of e2's elements are going to be maxes until it catches up
			while (i2.hasNext() && e1.index() > e2.index())
			{
				max += e2.get();
				e2 = i2.next();
			}
			// all of e1's elements are going to be maxes until it catches up
			if (e1.index() < e2.index())
			{
				max += e1.get();
			}
			// indices match, actually figure out the min and max
			if(e1.index() == e2.index())
			{
				if (e1.get() > e2.get())
				{
					min += e2.get();
					max += e1.get();
				}
				else
				{
					min += e1.get();
					max += e2.get();
				}
				if (i2.hasNext())
				{
					e2 = i2.next();
					if (!i1.hasNext())
						max += e2.get();
				}
				else
					break;
			}
		}

		// we ran out of i2, the rest of i1 is max
		while (i1.hasNext())
		{
			e1 = i1.next();		
			max += e1.get();
		}

		// ran out of i1, rest of i2 is max
		while (i2.hasNext())
		{
			max += e2.get();
			e2 = i2.next();
		}
		return min/max;
	}

	/**
	 * Compute the min max distance between two vectors considering only a subset of the features
	 * |v1| = |v2| = |features| must be true
	 * @param v1
	 * @param v2
	 * @param useFeature true at indices of the features we wish to consider
	 * @return minmax distance between v1 and v2 at features specified
	 */
	public static double minMaxSimilarity(Vector v1, Vector v2, boolean[] useFeature)
	{
		double min = 0;
		double max = 0;		

		Iterator<Element> i1 = v1.nonZeroes().iterator();
		Iterator<Element> i2 = v2.nonZeroes().iterator();

		Element e1 = null, e2 = null;

		// if a vector is 0, result will always be 0 
		if (!i1.hasNext() || !i2.hasNext())
			return 0;

		e2 = i2.next();
		while (i1.hasNext())
		{	
			e1 = i1.next();				
			// all of e2's elements are going to be maxes until it catches up
			while (i2.hasNext() && e1.index() > e2.index())
			{
				if (useFeature[e2.index()])
					max += e2.get();
				e2 = i2.next();
			}
			// all of e1's elements are going to be maxes until it catches up
			if (e1.index() < e2.index())
			{
				if (useFeature[e1.index()])
					max += e1.get();
			}
			// indices match, actually figure out the min and max
			if(e1.index() == e2.index())
			{
				if (useFeature[e1.index()])
				{
					if (e1.get() > e2.get())
					{
						min += e2.get();
						max += e1.get();
					}
					else
					{
						min += e1.get();
						max += e2.get();
					}
				}
				if (i2.hasNext())
				{
					e2 = i2.next();
					if (!i1.hasNext() && useFeature[e2.index()])
						max += e2.get();
				}
				else
					break;
			}
		}

		// we ran out of i2, the rest of i1 is max
		while (i1.hasNext())
		{
			e1 = i1.next();		
			if (useFeature[e1.index()])
				max += e1.get();
		}

		// ran out of i1, rest of i2 is max
		while (i2.hasNext())
		{
			if (useFeature[e2.index()])
				max += e2.get();
			e2 = i2.next();
		}
		return min/max;
	}

	public static Map<String, List<String>> findTopImposters(List<String> impostors, Map<String,Vector> vectorMap, String... posts) 
	{
		Map<String, List<String>> bestImpostersMap = new ConcurrentHashMap<>();
		// no impostors, exit
		if(impostors.isEmpty())
		{
			System.err.println("Why don't you have any impostors? ");
			System.exit(-1);
		}
//		// too few impostors to filter
		if (impostors.size() <= Parameters.IMPOSTER_COUNT)
		{	
			for(String post : posts)
				bestImpostersMap.put(post, impostors);
			return bestImpostersMap;
		}

		// choose top k impostors
		PriorityQueue<Pair<Double, String>> bestMatches;
		ArrayList<String> topImposters;
		double similarity;

		for(String key : posts)
		{
			Vector v1 = vectorMap.get(key);
			bestMatches = new PriorityQueue<>(Parameters.IMPOSTER_COUNT);

			for (String impostor : impostors)
			{
//               // this stuff only relevant for blog posts
//				String[] impAuthor = impostor.split(",");
//				if (key.split(",")[1].equals(impAuthor[1]))// || impAuthor[0].matches("\\d"))
//					continue;
				
				Vector v2 = vectorMap.get(impostor);
				similarity = minMaxSimilarity(v1, v2);

				if (bestMatches.size() < Parameters.IMPOSTER_COUNT) 
				{
					bestMatches.add(new Pair<Double, String>(similarity, impostor));
				}				
				else if(similarity > bestMatches.peek().getFirst())
				{
					bestMatches.poll().getFirst();
					bestMatches.add(new Pair<Double, String>(similarity, impostor));
				}				
			}
			
			topImposters = new ArrayList<>();
			for (Pair<Double, String> i : bestMatches)
			{
				topImposters.add(i.getSecond());
//				if (i.getSecond().split(",")[1].equals(key.split(",")[1]) || i.getSecond().split(",")[1].equals(key.split(",")[1]))
//					System.err.println("find top fucked");
			}

			bestImpostersMap.put(key, topImposters);	
		}
		return bestImpostersMap;
	}

	public static void main(String[] args)
	{
		Vector v1 = new SequentialAccessSparseVector(5);
		Vector v2 = new SequentialAccessSparseVector(5);
		
		for (int i = 0; i < 5; i++)
		{
			v1.set(i, i);
			v2.set(i, (i+1)*(i%2));
			System.out.println(v1.get(i) + "\t" + v2.get(i));
		}
		System.out.println(ImpostorUtilities.minMaxSimilarity(v1, v2));
		System.out.println();
		
		boolean[] f = new boolean[]{true,true,false, true,true};
		Vector v3 = new SequentialAccessSparseVector(5);
		Vector v4 = new SequentialAccessSparseVector(5);		
		for (int i = 0; i < 5; i++)
		{
			v3.set(i, (i+1)*(i%2));
			v4.set(i, i);
			System.out.println(v3.get(i) + "\t" + v4.get(i) +"\t" + f[i]);
		}
		System.out.println(ImpostorUtilities.minMaxSimilarity(v3, v4,f));

		Vector v5 = new SequentialAccessSparseVector(5);
		Vector v6 = new SequentialAccessSparseVector(5);
		
		v5.set(0,3);
		v6.set(0,3);
		System.out.println(v5.get(0) + "\t" + v6.get(0));
		for (int i = 1; i < 4; i++)
		{
			v5.set(i, (i+1)*(i%2));
			v6.set(i, i);
			System.out.println(v5.get(i) + "\t" + v6.get(i));
		}
		v5.set(4, 0);
		v6.set(4, 0);
		System.out.println(v5.get(4) + "\t" + v6.get(4));
		
		System.out.println(ImpostorUtilities.minMaxSimilarity(v5, v6));
	}

}
