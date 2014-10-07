package co.pemma.authorVerification;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.commons.io.FileUtils;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;

import co.pemma.authorVerification.Utilities;
import edu.drexel.psal.jstylo.generics.SimpleAPI;

public class WritePrintsVectors 
{
	static String TEMP_DIR = "/tmp/posts/";

	public static Map<String, Vector> jstyloVectors(Map<String, String> postData)
	{
		return jstyloVectors(postData, new ArrayList<>(postData.keySet()));
	}
	
	public static Map<String, Vector> jstyloVectors(Map<String, String> postData, List<String> clusterPosts)
	{
		// create / clear out temp directory
		try 
		{
			File file = new File(TEMP_DIR);
			if (file.exists())
				FileUtils.cleanDirectory(file);
			else
				file.mkdir();
		} 
		catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} 
		Map<String, Double> allUpper = new HashMap<>();
		Map<String, Double> allLower = new HashMap<>();
		Map<String, Double> firstUpper = new HashMap<>();
		Map<String, Double> camelCap = new HashMap<>();
		Map<String, Double> other = new HashMap<>();

		Map<String, double[]> yuleKs = new HashMap<>();

		// export posts to disk to make into vectors, i know...		
		for (String p : clusterPosts)
		{			
			String[] words = postData.get(p).split("\\s+");
			Map<String, Double> wordFrequencies = new HashMap<>();
			for (String word: words)
			{
				if (wordFrequencies.containsKey(word))
					wordFrequencies.put(word, wordFrequencies.get(word));
				else
					wordFrequencies.put(word, 1.);
			}
			yuleKs.put(p, calculateYuleK(wordFrequencies));
			getCaseCount(words, p, allLower, allUpper, firstUpper, camelCap, other);
			try(PrintWriter writer = new PrintWriter(TEMP_DIR+p))
			{
				writer.write(postData.get(p));
			} 
			catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}	
		System.out.println("Wrote " + clusterPosts.size() + " posts to file.");

		// make sylometry vectors for each post in clusterPosts and convert to mahout
		Map<String, double[]> doubleVectorMap = SimpleAPI.makeStyloVectors(TEMP_DIR, Utilities.MAX_THREADS);
		for (Entry<String, double[]> e : doubleVectorMap.entrySet())
		{
			double[] startVector = e.getValue();
			double[] withCase = new double[startVector.length + 16];
			int i;
			for (i = 0; i <  startVector.length; i++)
				withCase[i] = startVector[i];
			withCase[i++] = allLower.get(e.getKey());
			withCase[i++] = allUpper.get(e.getKey());
			withCase[i++] = firstUpper.get(e.getKey());
			withCase[i++] = camelCap.get(e.getKey());
			withCase[i++] = other.get(e.getKey());
			double[] legomenas = yuleKs.get(e.getKey());
			for (int j = 0; j < 11; j++)
			{
				if (legomenas.length > j)
					withCase[i++] = legomenas[j];
				else
					withCase[i++] = 0;
			}
		}
		System.out.println("Attempted to make stylo vectors for " + clusterPosts.size() + " posts...");
		System.out.println("Actually created " + doubleVectorMap.size() + " vectors.");

		return normalizeVectors(doubleVectorMap);		
	}

	private static void getCaseCount(String[] words, String p, Map<String, Double> allLower,
			Map<String, Double> allUpper, Map<String, Double> firstUpper,
			Map<String, Double> camelCap, Map<String, Double> other) 
	{
		double fu =0, au = 0, al = 0, o = 0, cc = 0;		
		for (String w : words)
		{
			if (w.matches("[A-Z][a-z]+"))
				fu++;
			else if (w.matches("[A-Z]+"))
				au++;
			else if (w.matches("[a-z]+"))
				al++;	
			else if (w.matches("[a-z]+[A-Z]+[a-z]+"))
				cc++;
			else
				o++;
		}
		allLower.put(p, al/words.length);
		allUpper.put(p, au/words.length);
		firstUpper.put(p, fu/words.length);
		camelCap.put(p, cc/words.length);
		other.put(p, o/words.length);
	}

	//calculate yules k as well as frequency of words that occur once, twice, ... n times
	private static double[] calculateYuleK(Map<String, Double> wordFrequencies) 
	{
		// m1 = number of different words in document (review)
		double m1 = wordFrequencies.keySet().size();

		Map<Double, Double> yuleMap = Utilities.reverseMap(wordFrequencies);

		// m2 = sum of products of each observed frequency squared and number of 
		// words with that frequency
		double m2 = 0.0;
		double[] legomenas = new double[11];
		for(Entry<Double,Double> e : yuleMap.entrySet()){
			Double freq = e.getKey();
			Double wordCount = e.getValue();
			m2 += freq*freq*wordCount;
			if (freq.intValue() < legomenas.length)
				legomenas[freq.intValue()] = wordCount;
		}
		legomenas[0] = 10000*(m2-m1)/(m1*m1); 
		return legomenas;
	}

	private static Map<String, Vector> normalizeVectors(Map<String, double[]> doubleVectorMap) 
	{
		double[] meanVector = null;
		double[] meanCounts = null;
		Map<String, Vector> vectors = new HashMap<>();

		for (Entry<String, double[]> e : doubleVectorMap.entrySet())
		{
			if (meanVector == null)
			{
				meanVector = new double[e.getValue().length];
				meanCounts = new double[e.getValue().length];
				for (int i = 0; i < meanVector.length; i++)
				{
					meanVector[i] = 0;
					meanCounts[i] = 0;
				}
			}
			Vector sparseVector = new RandomAccessSparseVector(e.getValue().length);
			sparseVector.assign(e.getValue());
			// sum values at each index to get mean
			for (Element el : sparseVector.nonZeroes())
			{
				meanVector[el.index()] += el.get();
				meanCounts[el.index()]++;
			}
			vectors.put(e.getKey(), sparseVector);
		}
		// divide mean vector by counts
		for (int i = 0; i < meanVector.length; i++)
		{
			meanVector[i] = meanVector[i]/meanCounts[i];
		}

		Vector v;
		// need to normalize stylo vectors, divide each element by max possible
		for(Entry<String, Vector> e : vectors.entrySet())
		{
			// row norm
			v = e.getValue();
			v = v.divide(v.norm(2));
			// feature mean nonzero
			for (Element el : e.getValue().nonZeroes())
			{
				el.set(el.get()/meanVector[el.index()]);
			}
		}
		return vectors;
	}	
}
