package co.pemma.authorVerification;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import org.apache.mahout.common.Pair;
import org.apache.mahout.math.Vector;

public class ImpostorSimilarity 
{
	/**
	 * @param test list of test keys
	 * @param train list of train keys
	 * @param impostors list of impostor keys
	 * @param vectorMap map from keys to vectors
	 * @param K how many iterations to run experiment
	 */
	public static double run(List<String> impostors, String x, String y, Map<String,Vector> vectorMap, int K)
	{
		ArrayList<String> filteredImpostors = new ArrayList<>();
		for (String i : impostors)			
			//			if (!(i.split(",")[1].equals(x.split(",")[1]) || i.split(",")[1].equals(y.split(",")[1])))
			filteredImpostors.add(i);

		//				List<String> xBest = bestImpostors(x, filteredImpostors, vectorMap);
		//				List<String> yBest = bestImpostors(y, filteredImpostors, vectorMap);
		//				return run(xBest, yBest, x, y, vectorMap, K);
		return getSimilarity(filteredImpostors, filteredImpostors, x, y, vectorMap, K);
	}

	public static double run(Map<String, List<String>> bestImpostorsMap, String x, String y, Map<String,Vector> vectorMap, int K)
	{
		List<String> xBest = bestImpostorsMap.get(x);
		List<String> yBest = bestImpostorsMap.get(y);
		return getSimilarity(xBest, yBest, x, y, vectorMap, K);
	}


	private static double getSimilarity(List<String> xBest, List<String> yBest, String x, String y, Map<String,Vector> vectorMap, int K) 
	{
		Vector xVector = vectorMap.get(x);
		Vector yVector = vectorMap.get(y);

		boolean[] randomFeatures;
		double score = 0;
		// choose random subset of the impostors to use
		for ( int k = 0; k < K; k++)
		{
			// select random subset of the best impostors of x and y		
			List<String> xImpostors = new ArrayList<>(Parameters.IMPOSTERS_PER_IT);
			List<String> yImpostors = new ArrayList<>(Parameters.IMPOSTERS_PER_IT);

			for (int index : ImpostorUtilities.randomIndices(xBest.size(), Parameters.IMPOSTERS_PER_IT))
				xImpostors.add(xBest.get(index));
			for (int index : ImpostorUtilities.randomIndices(yBest.size(), Parameters.IMPOSTERS_PER_IT))
				yImpostors.add(yBest.get(index));

			// choose random subset of the features to use
			randomFeatures = ImpostorUtilities.randomFeatures(xVector.size(), Parameters.FEATURE_SUBSET_PERCENT); 
			score += similarityIteration(xVector, yVector, xImpostors, yImpostors, randomFeatures, vectorMap, x, y);
		}
		// average the yScore and xScore up here by divide by 2
		score = (score/2.)/K;
		return score;
	}

	public static double similarityIteration(Vector xVector, Vector yVector, List<String> xImpostors,
			List<String> yImpostors, boolean[] randomFeatures, Map<String,Vector> vectorMap, String x, String y) 
	{
		// compare each test post to train posts and impostors for selected subsets
		double pairSim = ImpostorUtilities.minMaxSimilarity(xVector, yVector, randomFeatures);	
		double yImpSim = maxImpostorSimilarity(xVector, yImpostors, randomFeatures, vectorMap, x, y);
		double xImpSim = maxImpostorSimilarity(yVector, xImpostors, randomFeatures, vectorMap, x, y);

		double similarity = 0;
		if (pairSim > yImpSim)
			similarity ++;
		if (pairSim > xImpSim)
			similarity ++;

		return similarity;
	}

	public static double maxImpostorSimilarity(Vector v1, List<String> impostors, boolean[] randomFeatures,
			Map<String,Vector> vectorMap, String x, String y)
	{
		double similarity = 0, impostorSim = 0;
		Vector v2;
		String imp = "";

		for (String impostor : impostors)
		{
			v2 = vectorMap.get(impostor);
			similarity = ImpostorUtilities.minMaxSimilarity(v1, v2, randomFeatures);					

			// find best impostor similarity
			if (similarity >= impostorSim){
				impostorSim = similarity;
				imp = impostor;
			}				
		}
		//		
		return impostorSim;
	}	

	private static List<String> bestImpostors(String p, List<String> impostors, Map<String, Vector> vectorMap) 
	{
		PriorityQueue<Pair<Double, String>> bestMatches = new PriorityQueue<>(Parameters.IMPOSTER_COUNT);

		Vector v1 = vectorMap.get(p);

		for (String impostor : impostors)
		{
			Vector v2 = vectorMap.get(impostor);
			double similarity = ImpostorUtilities.minMaxSimilarity(v1, v2);

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

		ArrayList<String> topImposters = new ArrayList<>();
		for (Pair<Double, String> i : bestMatches)
		{
			topImposters.add(i.getSecond());
		}
		return topImposters;	
	}
}
