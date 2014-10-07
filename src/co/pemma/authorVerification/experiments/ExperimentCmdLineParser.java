package co.pemma.authorVerification.experiments;

import co.pemma.authorVerification.Parameters;
import co.pemma.authorVerification.Utilities;
import org.apache.commons.cli.*;

import java.util.Random;

public class ExperimentCmdLineParser 
{
	Options options, help;
	String classifier;
	String feature;
	String output;
	String type;
	String data;
	String city;
	Random rand;
	double threshold;
    double merge;
	boolean writeprints;
    boolean useBest;
	boolean ngrams;
	boolean tfidf;
	boolean lsh;
	int mistakenImpostors;
	int divisions;
	int trainSize;
    int clusterK;
    int authors;


	public ExperimentCmdLineParser(String[] args)
	{
		// set up defaults
		rand = new Random();
		output = null;
		lsh = false;
		ngrams = false;
		tfidf = false;
		writeprints = false;
        useBest = false;
		trainSize = 0;
		city = null;
		threshold = 1.0;
        merge = 1.0;
		mistakenImpostors = 0;
		divisions = 2;
        classifier = "";
        clusterK = 1;
		// define and parse args
		defineCmds();
		parseCmdLine(args);
	}
	
	@SuppressWarnings("static-access")
	public void defineCmds()
	{
		// create the different command line flags
		options = new Options();
		options.addOption(OptionBuilder.withLongOpt("help").create("h"));

		options.addOption(OptionBuilder.withArgName("impostorCount")
				.withLongOpt( "impCount" )
				.hasArg()
				.withDescription(  "total number of impostors to use." )
				.create( "ic"));
		options.addOption(OptionBuilder.withArgName("impostorsPerIteration")
				.withLongOpt( "impsPerIt" )
				.hasArg()
				.withDescription(  "Number of impostors to randomly select at each iteration" )
				.create( "ipi"));
		options.addOption(OptionBuilder.withArgName("featureSubset")
				.withLongOpt( "featureSubset" )
				.hasArgs()
				.withDescription(  "Percentage of features to randomly select at each iteration." )
				.create( "fs"));
		options.addOption(OptionBuilder.withArgName("K")
				.withLongOpt( "iterations" )
				.hasArg()
				.withDescription(  "Number of iterations to run each impostor similarity for" )
				.create( "k"));
		options.addOption(OptionBuilder.withArgName("output")
				.withLongOpt( "output" )
				.hasArg()
				.withDescription(  "Location to export results to" )
				.create( "o"));
		options.addOption(OptionBuilder.withArgName("lsh")
				.withLongOpt( "lsh" )
				.withDescription(  "Don't use locality sensitive hashing as preprocessing to clustering (use by default)" )
				.create( "lsh"));
		options.addOption(OptionBuilder.withArgName("type")
				.withLongOpt( "type" )
				.hasArg()
				.isRequired()
				.withDescription(  "Experiment type to run : 'cluster', 'pairs', 'learn-pairs', or 'learn-clusters'" )
				.create( "t"));
		options.addOption(OptionBuilder.withArgName("feature")
				.withLongOpt( "feature" )
				.hasArg()
//				.isRequired()
				.withDescription(  "Feature set to use: 'impostor', 'ngram', 'tfidf', 'stems', " +
                        "'compare' (runs all on same data), or 'allpairs (compares lsh to allpairs for ngram and impostor) " )
				.create( "f"));
		options.addOption(OptionBuilder.withArgName("data")
				.withLongOpt( "data" )
				.hasArg()
				.isRequired()
				.withDescription(  "Which dataset to use : 'backpage', 'blog', or 'aaac" )
				.create( "d"));
		options.addOption(OptionBuilder.withArgName("authors")
				.withLongOpt( "authorCount" )
				.hasArg()
				.withDescription(  "Number of authors to use in experiment" )
				.create( "ac"));
		options.addOption(OptionBuilder.withArgName("city")
				.withLongOpt( "city" )
				.hasArg()
				.withDescription(  "Restict backpage clustering to a single city" )
				.create( "cit"));
		options.addOption(OptionBuilder.withArgName("mistakenImpostors")
				.withLongOpt( "mistakenImpostors" )
				.hasArg()
				.withDescription(  "Insert mistaken impostors to test robustness" )
				.create( "mi"));
		options.addOption(OptionBuilder.withArgName("divisions")
				.withLongOpt( "divisions" )
				.hasArg()
				.withDescription(  "Minimum divisions for blog post length. Needed for mistaken imp experiments." )
				.create( "div"));
		options.addOption(OptionBuilder.withArgName("threshold")
				.withLongOpt( "threshold" )
				.hasArg()
				.withDescription(  "Similarity threshold cutoff for backpage clustering" )
				.create( "th"));
        options.addOption(OptionBuilder.withArgName("merge")
                .withLongOpt( "merge" )
                .hasArg()
                .withDescription(  "Similarity threshold for merging authors in truth labels" )
                .create( "merge"));
		// learning paramaters
		options.addOption(OptionBuilder.withArgName("classifier")
				.withLongOpt( "classifier" )
				.hasArg()
				.withDescription(  "Classifier to use to learn threshold : 'svm', 'regresssion', or 'bayes'" )
				.create( "c"));
        options.addOption(OptionBuilder.withArgName("clusterK")
                .withLongOpt( "clusterK" )
                .hasArg()
                .withDescription(  "Number of output clusters to produce." )
                .create( "ck"));
		options.addOption(OptionBuilder.withArgName("ngrams")
				.withLongOpt( "ngrams" )
				.withDescription(  " use ngram feature in classifer (off by default)" )
				.create( "ng"));
		options.addOption(OptionBuilder.withArgName("tfidf")
				.withLongOpt( "tfidf" )
				.withDescription(  " use tfidf feature in classifer (off by default)" )
				.create( "tfidf"));
		options.addOption(OptionBuilder.withArgName("writeprints")
				.withLongOpt( "writeprints" )
				.withDescription(  " use writeprints feature in classifer (off by default)" )
				.create( "wp"));
		options.addOption(OptionBuilder.withArgName("trainSize")
				.withLongOpt( "trainSize" )
				.hasArg()
				.withDescription(  "Number of pairs to use for training (1 = 1 postiive and 1 negative example)" )
				.create( "ts"));
        options.addOption(OptionBuilder.withArgName("useBest")
                .withLongOpt( "best" )
                .withDescription(  "Use the top (-ic) impostors for each post, default is sample at random." )
                .create( "best"));
		options.addOption(OptionBuilder.withArgName("singleThread")
				.withLongOpt( "singleThread" )
				.withDescription(  "Set to use single thread so cluster doesn't get mad and fail" )
				.create( "st"));
		options.addOption(OptionBuilder.withArgName("randomFile")
				.withLongOpt( "randomFile" )
				.withDescription(  "Append random number to end of output file to avoid concurrancy collisions" )
				.create( "rf"));

		// check for help first to avoid required field clash
		help = new Options();
		help.addOption(OptionBuilder.withLongOpt("help").create("h"));
	}

	public void parseCmdLine(String[] args)
	{
		HelpFormatter formatter = new HelpFormatter();

		try {
			CommandLine cl = new GnuParser().parse(help, args, true);		
			if (cl.hasOption("help"))
				formatter.printHelp( "Blog Experiment", options );

			// parse the input args
			else
			{
				CommandLine commandArgs = new GnuParser().parse( options, args );
				if( commandArgs.hasOption( "o" ) )
					output = commandArgs.getOptionValue("o").trim();
				if( commandArgs.hasOption( "rf" ) )
					output += rand.nextInt(1000000);
				if( commandArgs.hasOption( "t" ) )
					type = commandArgs.getOptionValue("t");
				if( commandArgs.hasOption( "f" ) )
					feature = commandArgs.getOptionValue("f");
				if( commandArgs.hasOption( "d" ) )
					data = commandArgs.getOptionValue("d");
				if( commandArgs.hasOption( "ac" ) )
					authors = Integer.parseInt(commandArgs.getOptionValue("ac"));
                if( commandArgs.hasOption( "ck" ) )
                    clusterK = Integer.parseInt(commandArgs.getOptionValue("ck"));
				if( commandArgs.hasOption( "lsh" ) )
					lsh = true;
				if( commandArgs.hasOption( "city" ) )
					city = commandArgs.getOptionValue("city").trim();
				if( commandArgs.hasOption( "mi" ) )
					mistakenImpostors = Integer.parseInt(commandArgs.getOptionValue("mi"));
				if( commandArgs.hasOption( "div" ) )
					divisions = Integer.parseInt(commandArgs.getOptionValue("div"));
				if( commandArgs.hasOption( "th" ) )
					threshold = Double.parseDouble(commandArgs.getOptionValue("th"));
                if( commandArgs.hasOption( "merge" ) )
                    merge = Double.parseDouble(commandArgs.getOptionValue("merge"));
				
				// choose classifier and the features to use (always use impostor similarity)
				if( commandArgs.hasOption( "c" ) )
					classifier = commandArgs.getOptionValue("c");
				if( commandArgs.hasOption( "ng" ) )
					ngrams = true;
				if( commandArgs.hasOption( "tfidf" ) )
					tfidf = true;
				if( commandArgs.hasOption( "wp" ) )
					writeprints = true;
                if( commandArgs.hasOption( "best" ) )
                    useBest = true;
				if( commandArgs.hasOption( "ts" ) )
					trainSize = Integer.parseInt(commandArgs.getOptionValue("ts"));
				
				if( commandArgs.hasOption( "st" ) )
					Utilities.MAX_THREADS = 1;

				if( commandArgs.hasOption( "ic" ) )
					Parameters.IMPOSTER_COUNT = Integer.parseInt(commandArgs.getOptionValue("ic"));
				if( commandArgs.hasOption( "ipi" ) )
					Parameters.IMPOSTERS_PER_IT = Integer.parseInt(commandArgs.getOptionValue("ipi"));
				if( commandArgs.hasOption( "fs" ) )
					Parameters.FEATURE_SUBSET_PERCENT = Double.parseDouble(commandArgs.getOptionValue("fs"));
				if( commandArgs.hasOption( "k" ) )
					Parameters.K = Integer.parseInt(commandArgs.getOptionValue("k"));
			}
		}
		catch( ParseException exp ) {
			formatter.printHelp( "", options );
			System.out.println( exp.getMessage() );
		}
	}
}
