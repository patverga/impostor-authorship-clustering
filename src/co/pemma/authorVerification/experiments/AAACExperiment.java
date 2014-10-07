package co.pemma.authorVerification.experiments;

import co.pemma.authorVerification.NGramVectors;
import co.pemma.authorVerification.Utilities;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class AAACExperiment extends Experiment
{
	String problemSet = "H";
	Map<String, List<String>> authorPostContentMap;

	public AAACExperiment()
	{
		super();
		DATA_LOCATION = "data/aaac";
		this.authorPostContentMap 	= new HashMap<>();
		readInData();
	}

    @Override
    protected void pairs(String output, NGramVectors.TokenType feature) {

    }

    @Override
    protected void mistakenImpostors(String output, int mistakenImpostors, int divisions) {

    }


    @Override
	protected Map<String, String> setupClusterData(Map<String, Integer> labels, List<String> clusterPosts) 
	{
		System.out.print("Randomly choosing test data and imposters...");
		Map<String,String> posts = new HashMap<>();
		int authorNumber;
		String key;
		

		List<String> authorPosts;
		// randomly choose authors to attempt to cluster
		for (Entry<String, List<String>> e : authorPostContentMap.entrySet())
		{
			key = e.getKey();
			authorNumber = Integer.parseInt(key.replaceAll("\\D", ""));
			authorPosts = e.getValue();

			for (int i = 0; i < authorPosts.size(); i++)
			{
				if (key.startsWith(problemSet))
				{
					clusterPosts.add(i + "," + key);
					labels.put(i + "," + key, authorNumber);
				}
				else
					impostors.add(i + "," + key);
				posts.put(i + "," + key, authorPosts.get(i));
			}
		}	

		System.out.println(" Done");
		System.out.println("Found " + impostors.size() + " imposters.");
		System.out.println(posts.size() + " total posts.");
		return posts;
	}

	public void readInData()
	{
		System.out.println("Reading in blog files...");
		final File directory = new File(DATA_LOCATION);

		ExecutorService executor = Executors.newFixedThreadPool(Utilities.MAX_THREADS);		
		for (final File problem : directory.listFiles()) 
		{
			for (final File file : problem.listFiles()) 
			{
				executor.submit(new ReadInDataThread(file, problem));
			}
		}
		Utilities.waitForThreads(executor);
		System.out.println("Found " + authorPostContentMap.size() + " authors.");
	}
	
	public class ReadInDataThread implements Runnable 
	{
		private File file;
		private File parent;

		ReadInDataThread(File file, File parent)
		{
			this.file = file;
			this.parent = parent;
		}

		@Override
		public void run() 
		{
			List<String> authorPostList;
			String line, text, author;

			try (BufferedReader reader = new BufferedReader(new FileReader(file)))
			{
				author = file.getName();
				if (author.contains("sample"))
					author = author.substring(7, 9);
				if (author.contains("train"))
					author = author.substring(6, 8);
				author = parent.getName().replaceAll("problem", "") + author;
				text = "";				
				while ((line = reader.readLine()) != null)
				{
					if ( !line.startsWith("Name Name") && !line.startsWith("Essay ") 
							&& !line.startsWith("Paper ") )
						text += line;					
				}				
				if (authorPostContentMap.containsKey(author))
					authorPostList = authorPostContentMap.get(author);
				else
				{
					authorPostList = new ArrayList<>();
				}
				authorPostList.add(text);
				authorPostContentMap.put(author, authorPostList);					
			} 
			catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}		
		}
	}
}
