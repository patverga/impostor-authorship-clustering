package co.pemma.authorVerification;

public class Parameters 
{

	// how many top impostors to choose for each test and train post
	public static int IMPOSTER_COUNT = 250;
	// how many impostors to use per iteration
	public static int IMPOSTERS_PER_IT = 25;
	// percent of features to randomly choose per iteration
	public static double FEATURE_SUBSET_PERCENT = 0.4;
	// iterations to run imposter experiment for
	public static int K = 100;
	// number of bins for thresholding and output
	public static int BINS = K * 10;
	
	public static String paramsToString()
	{
		return (IMPOSTER_COUNT + "\t" + IMPOSTERS_PER_IT + "\t" + FEATURE_SUBSET_PERCENT + "\t" + K);
	}	
}
