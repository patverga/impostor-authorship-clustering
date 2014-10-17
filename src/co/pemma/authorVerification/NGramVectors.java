package co.pemma.authorVerification;


import cc.factorie.app.strings.PorterStemmer;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.javatuples.Pair;

import java.util.*;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;


public class NGramVectors 
{
	// size of ngram window (only supports a single size ngram currently)
	final static int N_GRAM_SIZE = 4;
	// how many documents must ngram occur in to be considered
	final static int MIN_DOC_OCCURANCES = 1;
	// only use most frequent ngrams (use really big number to disable)
	final static int TOP_NGRAM_K = 100000; //100k
	// convert all ngrams to lower case
	final static boolean ALL_LOWER = false;
	// use iverse document frequency
	final static boolean IDF = true;
	// return a concurrent hash map, false returns regular hashmap
	final static boolean THREAD_SAFE = true;


	public static Map<String, Vector> run(Map<String, String> inputPosts)
	{
		return run(inputPosts, TokenType.NGRAMS);
	}

	public static Map<String, Vector> run(Map<String, String> inputPosts, TokenType token)
	{
		System.out.println("Extracting size " + N_GRAM_SIZE + " ngrams from " + inputPosts.size() + " Documents...");
		// each doc key maps to a list of its ngrams
		Map<String, List<String>> documentNgrams = extractNGrams(inputPosts, token);

		// this is a little gross
		// how many times each ngram occurs in total in the corpus	// how many documents each ngram occurs in
		Pair<Map<String, Double>, Map<String, Double>> frequencies = getTotalFrequencyCounts(documentNgrams);		
		Map<String, Double> totalFrequencies = frequencies.getValue0();		
		Map<String, Double> documentOccurances = frequencies.getValue1();

		System.out.println("Keeping top ngrams...");
		PriorityQueue<Pair<Double, String>> chosenNgrams = topNGrams(totalFrequencies, documentOccurances);

		System.out.print("Creating vectors");
		Map<String, Integer> nGramIndices = nGramsToIndices(chosenNgrams);
		Map<String, Vector> vectors = createVectors(nGramIndices, documentNgrams, documentOccurances, inputPosts.size());

		return vectors;
	}

	public static Map<String, List<String>> extractNGrams(Map<String, String> inputTexts, TokenType token)
	{
		Map<String, List<String>> documentNgrams = new HashMap<>();
		String text;
		int iteration = 0;
		for (Entry<String, String> document : inputTexts.entrySet())
		{
			text = document.getValue();
			if (ALL_LOWER)
				text = text.toLowerCase();

			String[] words = text.split("\\s+");
			List<String> ngrams = extractDocumentTerms(words, token);
			documentNgrams.put(document.getKey(), ngrams);
			Utilities.printPercentProgress(iteration++, inputTexts.size());
		}
		return documentNgrams;
	}

	private static List<String> extractDocumentTerms(String[] words, TokenType token)
	{
		List<String> ngramsFound = new ArrayList<>();

		for (String word : words)
		{
			// we are only caring about tf-idf
			if (token == TokenType.TERMS)
				ngramsFound.add(word);			
			else if(token == TokenType.STEMS)
			{
				String s = PorterStemmer.apply(word);
				ngramsFound.add(s);
			}
			// if ngram is small enough just add it 
			else if (word.length() <= N_GRAM_SIZE)
			{
				ngramsFound.add(word);
			}
			// else do sliding window over each NGRAMSIZE block of the word
			else
			{
				for (int i = 0; i <= word.length()-N_GRAM_SIZE; i++)
				{
					ngramsFound.add(word.substring(i, i+N_GRAM_SIZE));						
				}
			}
		}
		return ngramsFound;
	}

	/**
	 * 
	 * @param docNgrams map of each documents key to a list of its ngrams
	 * @return [0] the total frequency count of each ngram, [1] the number of documents each ngram occurs in
	 */
	private static Pair<Map<String, Double>, Map<String, Double>> getTotalFrequencyCounts(Map<String,List<String>> docNgrams) 
	{
		// how many times each ngram occurs in total in the corpus
		Map<String, Double> totalFrequencies = new HashMap<>();	
		// how many documents each ngram occurs in
		Map<String, Double> documentOccurances = new HashMap<>();

		for (List<String> ngrams : docNgrams.values())
		{
			Set<String> resultSet = new HashSet<>();
			for (String ngram : ngrams)
			{
				// how many times each ngram occurs overall in dataset
				if (totalFrequencies.containsKey(ngram))
					totalFrequencies.put(ngram, totalFrequencies.get(ngram) + 1);
				else
					totalFrequencies.put(ngram, 1.);
				// how many documents ngram occurs in -only add once for this document
				if (!resultSet.contains(ngram))
				{
					if (documentOccurances.containsKey(ngram))
						documentOccurances.put(ngram, documentOccurances.get(ngram) + 1);
					else
						documentOccurances.put(ngram, 1.);
					resultSet.add(ngram);
				}
			}
		}
		return new Pair<>(totalFrequencies,documentOccurances);
	}	

	/**
	 * select and keep only the TOP_NGRAM_K ngrams
	 * @return 
	 */
	private static PriorityQueue<Pair<Double, String>> topNGrams(Map<String, Double> totalFrequencies, Map<String, Double> documentOccurances)
	{
		// use priority queue for heap sort
		PriorityQueue<Pair<Double, String>> topNGrams = new PriorityQueue<>(TOP_NGRAM_K);
		String ngram;
		double occurances;

		int size = documentOccurances.size();
		// look at how many times each ngram occured
		for (Entry<String, Double> e : totalFrequencies.entrySet())
		{
			ngram = e.getKey();
			occurances = e.getValue(); 
			// if it did not occur in atleast MIN DOC documents, throw it out
			if (documentOccurances.get(ngram) >= MIN_DOC_OCCURANCES)
			{			
				// if heap is not full, toss the ngram in
				if (topNGrams.size() < TOP_NGRAM_K) 
				{
					topNGrams.add(new Pair<Double, String>(occurances, ngram));
				}				
				// if this ngram occurred enough to be in heap, put it in, remove lowest value ngram
				else if(occurances > topNGrams.peek().getValue0())
				{
					topNGrams.poll().getValue1();
					topNGrams.add(new Pair<Double, String>(occurances, ngram));
				}
			}			
		}		
		System.out.println("Keping at most top " + TOP_NGRAM_K + " ngrams out of " + size);
		return topNGrams;
	}

	/**
	 * map the ngrams we're keeping to indices
	 * @return the ngram-index map
	 */
	private static Map<String, Integer> nGramsToIndices(PriorityQueue<Pair<Double, String>> nGrams)
	{
		// map full ngram set to common indices
		Map<String, Integer> nGramIndices = new HashMap<>();
		int index = 0;
		for ( Pair<Double, String> ngram : nGrams)
			nGramIndices.put(ngram.getValue1(), index++);
		return nGramIndices;
	}

	public static Map<String, Vector> createVectors(Map<String, Integer> nGramIndices, 
			Map<String, List<String>> documentNGrams, Map<String, Double> documentOccurances, int size)
			{			
		System.out.print(" size " + nGramIndices.size() + "...");

		Map<String, Vector> vectors;
		if (THREAD_SAFE){
			vectors = new HashMap<>();
		}else{
			vectors = new ConcurrentHashMap<>();
		}

		for (Entry<String, List<String>> document : documentNGrams.entrySet())
		{			
			Map<String, Double> postCounts = new HashMap<>();

			// figure out the term frequency counts of each ngram that occurred in this post
			for (String ngram : document.getValue())
			{
				if (postCounts.containsKey(ngram))
					postCounts.put(ngram, postCounts.get(ngram) + 1);
				else
					postCounts.put(ngram, 1.);
			}		

			Vector vector = getTFIDFVector(nGramIndices, postCounts, documentNGrams, documentOccurances, size);

			vectors.put(document.getKey(), vector);
		}
		System.out.println("Done");
		return vectors;
			}

	private static Vector getTFIDFVector(Map<String, Integer> nGramIndices,Map<String, Double> postCounts,
			Map<String, List<String>> documentNGrams, Map<String, Double> documentOccurances, int size) 
	{
		Vector vector = new SequentialAccessSparseVector(nGramIndices.size());
		double tf, idf;

		// for each ngram that occurred in the post, get its tfidf
		for (Entry<String, Double> ngramFreq : postCounts.entrySet())
		{
			String ngram = ngramFreq.getKey();
			// only use ngrams that were not culled by minimum occurrence requirements
			if (nGramIndices.containsKey(ngram))
			{
				tf = ngramFreq.getValue();///postNGrams.size();
				if (IDF)
					// multiply by inverse document frequency
					idf = Math.log( size / (1+documentOccurances.get(ngram)) );
				else
					idf=1;
				vector.set(nGramIndices.get(ngram), (tf*idf) );
			}
		}
		return vector;
	}

	public enum TokenType{TERMS,NGRAMS,STEMS,IMPOSTOR,STYLO};

	public static void main(String[] args)
	{
		Map<String,String> posts = new HashMap<>();
		String a1 = "a1";
		String t1 = "test testing this ";

		String a2 = "a2";
		String t2 ="this is only a test";

		posts.put(a1, t1);
		posts.put(a2, t2);

		NGramVectors.run(posts);
	}
}
