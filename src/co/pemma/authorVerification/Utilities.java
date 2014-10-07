package co.pemma.authorVerification;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.Socket;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;


public class Utilities {

    public static int MAX_THREADS =  Runtime.getRuntime().availableProcessors()-1;

    public static void waitForThreads(ExecutorService executor)
    {
        // wait for thread pool shutdown before returning
        executor.shutdown();
        try {
            executor.awaitTermination(10, TimeUnit.DAYS);
        } catch (InterruptedException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    /**
	 * Creates a new Map associating values from the given keySet with the given
	 * default value.
	 * 
	 * @param keySet the keyset with which to initialize the new Map
	 * @param defaultValue default value to associate with the given keys
	 * @return new Map mapping the given keyset to the given default value
	 */
	public static <K,V> Map<K,V> newMapFromKeySet(Set<K> keySet, V defaultValue) {
		Map<K,V> newMap  = new HashMap<>(keySet.size());
		for(K key : keySet){
			newMap.put(key, defaultValue);
		}
		return newMap;
	}

	public static <T> Map<Double,Double> reverseMap(Map<T,Double> m) {
		Map<Double,Double> newMap  = new HashMap<>(m.size());
		for(Double value : m.values()){
			if(newMap.containsKey(value))
				newMap.put(value, newMap.get(value) + 1.0);
			else
				newMap.put(value, 1.0);
		}
		return newMap;
	}

	public static BufferedReader processSentence(String inputSentence) throws IOException{

		//		try(Socket connection = new Socket("localhost", 3228)){
		//			PrintWriter writer = new PrintWriter(new BufferedWriter(new OutputStreamWriter(connection.getOutputStream())), true);
		//			writer.println(inputSentence);
		//			connection.shutdownOutput();
		//			return new BufferedReader(new InputStreamReader(connection.getInputStream()));	
		//		} catch (IOException e){
		//			e.printStackTrace();
		//		}
		//		return null;

		// TODO fix socket closing situation
		// ok, actually, this socket is definitely always getting closed, Eclipse just doens't know it
		Socket connection = new Socket("localhost", 3228);
		PrintWriter writer = new PrintWriter(new BufferedWriter(new OutputStreamWriter(connection.getOutputStream())), true);
		writer.println(inputSentence);
		connection.shutdownOutput();
		return new BufferedReader(new InputStreamReader(connection.getInputStream()));	
	}

	/**
	 * prints the percentage complete in nice format for i of k iterations
	 * @param iteration current iteration
	 * @param total number of iterations
	 * @return incremented iteration
	 */
	public static int printPercentProgress(int iteration, double total)
	{
		if (++iteration == total)
			System.out.println(" Done ");
		else if (total >= 100)
		{
			total = Math.floor(total / 100);
			if ( (iteration % total) == 0 )
			{	
				if (iteration / total % 10 == 0)
					System.out.printf("%.0f",(int)iteration / total);
				else
					System.out.print(".");				
			}
		}
		else
		{
			int percentComplete = (int) (100*(iteration/total));
			for (int i = 0; i < iteration/total; i ++)
				System.out.print(".");			
			if (percentComplete % 10 == 0)
				System.out.printf("%d", percentComplete);
			else
				System.out.print(".");	
		}
		return iteration;
	}


	public static String readFile(File file)
	{
		String line, text = "";
		try (BufferedReader reader = new BufferedReader(new FileReader(file)))
		{
			while ((line = reader.readLine()) != null)
				text += line;	            
		} 
		catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return null;
		}
		return text;
	}
}
