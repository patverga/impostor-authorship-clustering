package co.pemma.authorVerification.experiments;

import co.pemma.authorVerification.*;
import co.pemma.authorVerification.NGramVectors.TokenType;
import co.pemma.authorVerification.Utilities;
import com.google.common.base.Joiner;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomDataGenerator;
import org.apache.commons.math3.util.ArithmeticUtils;
import org.apache.mahout.common.distance.CosineDistanceMeasure;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Vector;
import org.javatuples.Triplet;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.io.*;
import java.util.*;
import java.util.Map.Entry;
import java.util.regex.Pattern;

public class BlogExperiment extends Experiment
{
    // max number of blogs to read in
    private final int NUM_AUTHORS = 5000;
    private final int SNIPPET_LENGTH = 500;
    private final int POST_DIVISIONS;
    private final int MISTAKEN_IMPOSTORS;
    private final boolean ZIFF_POSTS = true;

    public BlogExperiment()
    {
        this(0,2);
    }

    public BlogExperiment(int mistakenImpostors, int divisions)
    {
        super();
        DATA_LOCATION = "data/blogs";
        this.POST_DIVISIONS = divisions;
        this.MISTAKEN_IMPOSTORS = mistakenImpostors;
    }

    /**
     * run pair experiment from original impostor paper
     */
    @Override
    public void pairs(String outputLocation, TokenType feature)
    {
        reset();
        List<Triplet<Double, Double, Integer>> results = setUpAndRunWinterData(feature);
        List<String> outputLines = pairResults(results, feature.toString());
        exportResultsToFile(outputLines, outputLocation);
    }

    private List<Triplet<Double, Double, Integer>> setUpAndRunWinterData(TokenType feature)
    {
        Map<String, String> begins = parseWinterDir("data/winter/Begin/Text/");
        Map<String, String> ends = parseWinterDir("data/winter/End/Text/");
        Map<String, String> impMap = parseWinterDir("data/winter/Decoy/Text/");

        List<String> keys = new ArrayList<>(begins.keySet());
        Collections.shuffle(keys);
        List<Triplet<String,String,Integer>> testPairs =  setupPairData(keys, testSize);

        Map<String, String> postData = new HashMap<>();

        for (Entry<String, String> e : begins.entrySet())
        {
            // get first post by author 1
            postData.put("f,"+e.getKey(), e.getValue());
        }
        for (Entry<String, String> e : ends.entrySet())
        {
            // get last post by author 2
            postData.put("l,"+e.getKey(), e.getValue());
        }
        for (Entry<String,String> e : impMap.entrySet())
        {
            postData.put(e.getKey(), e.getValue());
        }

        List<String> impList = new ArrayList<>(impMap.keySet());

        DistanceMeasure cosine = null;
        Map<String, Vector> vectors;
        switch(feature)
        {
            case IMPOSTOR:
                vectors = NGramVectors.run(postData, TokenType.NGRAMS);
                break;
            case STYLO:
                cosine = new CosineDistanceMeasure();
                vectors =  WritePrintsVectors.jstyloVectors(postData);
                break;
            default:
                vectors = NGramVectors.run(postData, feature);
                break;
        }
        return runWinterPairs(feature, testPairs, impList, cosine, vectors);
    }

    /**
     *
     * @param feature
     * @param testPairs
     * @param impList
     * @param cosine
     * @param vectors
     * @return [0] disimilaity of pair [1] 1=same pair, 0=diff pair
     */
    private List<Triplet<Double, Double, Integer>> runWinterPairs(TokenType feature, List<Triplet<String, String, Integer>> testPairs,
                                                      List<String> impList, DistanceMeasure cosine, Map<String, Vector> vectors)
    {
        int featureCount = vectors.get(impList.get(0)).size();

        // set up randoms
        boolean[][] randFeatures = new boolean[Parameters.K][featureCount];
        List<Integer>[] randImpostors = new List[Parameters.K];
        for (int i = 0; i < Parameters.K; i++)
        {
            randFeatures[i] = ImpostorUtilities.randomFeatures(featureCount, Parameters.FEATURE_SUBSET_PERCENT);
            randImpostors[i] = ImpostorUtilities.randomIndices(impList.size(), Parameters.IMPOSTERS_PER_IT);
        }

        // winter pair experiment
        double similarity;
        int iteration = 0;

        System.out.println("Testing Pairs");
        List<Triplet<Double, Double, Integer>> results = new ArrayList<>();
        for (Triplet<String, String, Integer> pair : testPairs)
        {
            String a1 = "f,"+pair.getValue0();
            String a2 = "l,"+pair.getValue1();

            Vector v1 = vectors.get(a1);
            Vector v2 = vectors.get(a2);

            similarity = 0;
            switch(feature)
            {
                case IMPOSTOR:

                    for (int i = 0; i < Parameters.K; i++)
                    {
                        List<String> imps = new ArrayList<>(Parameters.IMPOSTERS_PER_IT);
                        for (int index : randImpostors[i])
                            imps.add(impList.get(index));
                        similarity += ImpostorSimilarity.similarityIteration(v1, v2, imps, imps, randFeatures[i], vectors, a1, a2);
                    }
                    similarity = (similarity/2.)/Parameters.K;
                    break;

                case STYLO:
                    if (v1 != null && v2 != null)
                        similarity = 1 - cosine.distance(v1, v2);
                    break;
                default:
                    similarity = ImpostorUtilities.minMaxSimilarity(v1, v2);
                    break;
            }
            double minmaxSim = 1-ImpostorUtilities.minMaxSimilarity(v1,v2);

            results.add(new Triplet<>(1-similarity, minmaxSim, pair.getValue2()));
            updateResultCounts(similarity, pair.getValue2(), sameAuthor, diffAuthor);
            Utilities.printPercentProgress(iteration++, testPairs.size());
        }
        return results;
    }

    private Map<String, String> parseWinterDir(String dir)
    {
        Map<String, String> posts = new HashMap<>();
        File[] postFiles = new File(dir).listFiles();
        for (File f : postFiles)
        {
            StringBuilder post = new StringBuilder();
            try(BufferedReader reader = new BufferedReader(new FileReader(f)))
            {
                String line;
                while ((line = reader.readLine()) != null)
                {
                    line = line.trim();
                    post.append(line);
                }
            }
            catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
            posts.put(f.getName(), post.toString());
        }
        return posts;
    }

    private Map<String, String> getPostTextData(Map<String, List<String>> authorPosts)
    {
        Map<String, String> postData = new HashMap<>();

        for (Entry<String, List<String>> e : authorPosts.entrySet())
        {
            List<String> posts = e.getValue();

            // get first post by author 1
            postData.put("f,"+e.getKey(), posts.get(0));
            // get last post by author 2
            postData.put("l,"+e.getKey(), posts.get(posts.size()-1));

            // testing if they put all the data in to the vector creation machine
            //			for (int i = 1; i < posts.size()-1; i++)
            //				postData.put(i+","+e.getKey(), posts.get(i));
        }
        return postData;
    }

    private Map<String, String> getPostTextData(Map<String, List<String>> authorPosts,
                                                List<Triplet<String, String, Integer>> testPairs)
    {
        Map<String, String> postData = new HashMap<>();

        for (Triplet<String,String,Integer> t : testPairs)
        {
            String a1 = t.getValue0();
            String a2 = t.getValue1();

            // get first post by author 1
            postData.put("f,"+a1, authorPosts.get(a1).get(0));
            // get last post by author 2
            postData.put("l,"+a2, authorPosts.get(a2).get(authorPosts.get(a2).size()-1));

        }
        return postData;
    }

    /**
     * Randomly select authors and parse their posts two (first and last) snippets.
     *
     * @param blogDataPath
     * @return a map from authors (identified by filename) to lists of snippets
     */
    private Map<String, List<String>> readInData(final String blogDataPath, final int snippetLength,
                                                 final int numAuthors, boolean firstLast)
    {
        Map<String, List<String>> authorContentMap = new HashMap<>();
        System.out.println("Reading in blog files...");
        File directory = new File(blogDataPath);
        File[] blogFiles = directory.listFiles();
        File f;
        int i = 0;
        List<String> snippets;

        for (int dex : new RandomDataGenerator(new MersenneTwister(0)).nextPermutation(blogFiles.length, blogFiles.length))
        {
            if (authorContentMap.size() == numAuthors)
                break;
            f = blogFiles[dex];
            List<List<String>> parsedBlog =  parseBlogFile(f);
            // we only want the first and last 500 words
            if (firstLast)
            {
                List<String> words = flattenBlog(parsedBlog);
                if (words.size() < (POST_DIVISIONS * snippetLength)) {
                    continue;
                }
                snippets = getFirstLastSnippets(words, snippetLength);
            }
            // we want all 500 word chunks
            else
            {
                snippets = splitContentToSnippets(parsedBlog);
                if (snippets.size() < POST_DIVISIONS)
                    continue;
            }
            authorContentMap.put(f.getName(), snippets);
            Utilities.printPercentProgress(i++, blogFiles.length);
        }

        System.out.println("Found " + authorContentMap.size() + " authors.");
        return authorContentMap;
    }

    /**
     * concat all the posts into equal length snippets, each post appears in at most one snippet
     * @param parsedBlog all the se
     * @return
     */
    private List<String> splitContentToSnippets(List<List<String>> parsedBlog)
    {
        List<String> snippets = new ArrayList<>();
        StringBuilder text = new StringBuilder();
        int postIndex;
        int wordCount = 0;
        for(List<String> words : parsedBlog)
        {
            postIndex = 0;
            while(wordCount < SNIPPET_LENGTH && postIndex < words.size())
            {
                text.append(words.get(postIndex++));
                text.append(" ");
                wordCount++;
            }
            if (wordCount == SNIPPET_LENGTH)
            {
                wordCount = 0;
                snippets.add(text.toString());
                text = new StringBuilder();
            }
        }
        return snippets;
    }

    /**
     * @param rawBlog
     * @return a List of parsed posts; each parsed post is a List of Strings.
     */
    private List<List<String>> parseBlogFile(final File rawBlog)
    {
        final Pattern ignoredLinePattern = Pattern.compile("^(<Blog>)|(</Blog>)|(<date>)|(</post>)");
        List<List<String>> parsedBlog = new ArrayList<List<String>>();
        List<String> parsedPost = new ArrayList<String>();
        try(BufferedReader reader = new BufferedReader(new FileReader(rawBlog)))
        {
            String line;
            while ((line = reader.readLine()) != null)
            {
                line = line.trim();
                if (ignoredLinePattern.matcher(line).find())
                {
                    continue;
                }
                else if (line.startsWith("<post>"))
                {
                    if (parsedPost != null && parsedPost.size() > 0) {
                        parsedBlog.add(parsedPost);
                    }
                    parsedPost = new ArrayList<String>();
                }
                else
                {
                    Collections.addAll(parsedPost, line.replaceAll("urlLink|&nbsp", " ").split("\\s+"));
                }
            }
        }
        catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return parsedBlog;
    }

    /**
     *
     * @param parsedBlog a List of parsed posts; each parsed post is a List of Strings
     * @return a flat List of the words in all posts
     */
    private List<String> flattenBlog(final List<List<String>> parsedBlog) {
        int size = 0;
        for (List<String> parsedPost : parsedBlog) {
            size += parsedPost.size();
        }
        List<String> allWords = new ArrayList<String>(size);
        for (List<String> parsedPost : parsedBlog) {
            allWords.addAll(parsedPost);
        }
        return allWords;
    }

    /**
     *
     * @param words
     * @param snippetLength
     * @return snippets representing the first and last snippetLength words
     */
    private List<String> getFirstLastSnippets(final List<String> words, final int snippetLength){
        assert words.size() >= snippetLength * POST_DIVISIONS;
        List<String> firstLastSnippets = new ArrayList<String>(2);
        Joiner joiner = Joiner.on(" ");
        firstLastSnippets.add(joiner.join(words.subList(0, snippetLength)));
        firstLastSnippets.add(joiner.join(words.subList(words.size() - snippetLength, words.size())));
        return firstLastSnippets;
    }

    /**
     * @param keys author keys to choose from
     * @return [0] author1 [1] author2 [2] 1 if author1 and author2 are the same
     */
    private List<Triplet<String, String, Integer>> setupPairData(List<String> keys, int size)
    {
        List<Triplet<String, String, Integer>> pairs = new ArrayList<>();
        String author1;
        String author2;
        String author3;

        // choose the pairs we want to test
        int i = 0;
        while (pairs.size() < size*2 && i < keys.size()-3)
        {
            author1 = keys.get(i++);
            author2 = keys.get(i++);
            author3 = keys.get(i++);
            pairs.add(new Triplet<>(author1, author1, 1));
            pairs.add(new Triplet<>(author2, author3, 0));
        }
        return pairs;
    }


    private void printPairResults(String output, String feature)
    {
        List<String> results = printResults();
        if (output != null)
        {
            try(PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(output, true))))
            {
                for (String result : results)
                {
                    //					if (MISTAKEN_IMPOSTORS > 0)
                    writer.println(result + "\t" + MISTAKEN_IMPOSTORS + "\t" + POST_DIVISIONS + "\t" + feature);
                    //					else
                    //						writer.println(result + "\t impostor");
                }
            }
            catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
    }


    /**
     * learn pairs is dumb, but was used to show that it was dumb
     */
    @Override
    public void learnPairs(String output, String classifierType, boolean ngrams, boolean tfidf, boolean writeprints)
    {
        Map<String, List<String>> authorPostContentMap = readInData(DATA_LOCATION,SNIPPET_LENGTH, NUM_AUTHORS, false);

        List<String> keys = new ArrayList<>(authorPostContentMap.keySet());
        Map<String, String> postData = new HashMap<>();
        Classifier classifier = new Classifier(classifierType);

        // set up test data
        List<Triplet<String, String, Integer>> testPairs = setupPairData(keys.subList(0, testSize*3), testSize);
        for (Triplet<String, String, Integer> pair : testPairs)
        {
            String s1 = pair.getValue0();
            String s2 = pair.getValue1();

            List<String> l1 = authorPostContentMap.get(s1);
            List<String> l2 = authorPostContentMap.get(s2);

            postData.put("f,"+s1, l1.get(0));
            postData.put("l,"+s2, l2.get(l2.size()-1));
        }
        // set up training data
        List<Triplet<String, String, Integer>> trainPairs = setupPairData(keys.subList(testSize*3, keys.size()), trainSize);
        for (Triplet<String, String, Integer> pair : trainPairs)
        {
            String s1 = pair.getValue0();
            String s2 = pair.getValue1();

            List<String> l1 = authorPostContentMap.get(s1);
            List<String> l2 = authorPostContentMap.get(s2);

            postData.put("f,"+s1, l1.get(0));
            postData.put("l,"+s2, l2.get(l2.size()-1));
        }
        // get imposters
        generateImpostors(authorPostContentMap, postData, keys);

        // generate the vectors for the problem set and imposters
        Map<String, Vector> ngramVectors = NGramVectors.run(postData,TokenType.NGRAMS);
        Map<String, Vector> tfidfVectors = null;
        Map<String, Vector> writeprintsVectors = null;

        // train model
        ArrayList<Attribute> features = trainPairs(trainPairs, ngramVectors, tfidfVectors, writeprintsVectors, ngrams, tfidf, writeprints, postData, classifier);

        // test each pair
        System.out.println("Testing pairs...");
        Instances testSet = new Instances("test", features, testPairs.size());
        testSet.setClassIndex(testSet.numAttributes()-1);

        List<Triplet<Double, Double, Integer>> results = new ArrayList<>();
        int iteration = 0;
        for (Triplet<String, String, Integer> pair : testPairs)
        {
            Instance in = similarityVector("f,"+pair.getValue0(), "l,"+pair.getValue1(), features, ngrams, ngramVectors, tfidfVectors, writeprintsVectors);
            if (in != null)
            {
                in.setDataset(testSet);
                in.setClassValue("" + pair.getValue2());
                testSet.add(in);
                results.add(new Triplet<>(1 - classifier.classifyVector(in), 1., pair.getValue2()));
            }
            Utilities.printPercentProgress(iteration++, testPairs.size());
        }
//        classifier.batchTest(testSet);


        String learningParams = "\t"+classifierType+"\t"+ngrams+"\t"+tfidf+"\t"+writeprints+"\t"+trainSize;

        List<String> outputLines = pairResults(results, learningParams);
        exportResultsToFile(outputLines, output);
    }

    @Override
    public void mistakenImpostors(String output, int mistakenImpCount, int divisions)
    {
        Map<String, List<String>> authorPostContentMap = readInData(DATA_LOCATION,SNIPPET_LENGTH, NUM_AUTHORS, false);

        List<String> keys = new ArrayList<>(authorPostContentMap.keySet());
        Map<String, String> postData = new HashMap<>();

        // set up test data
        List<Triplet<String, String, Integer>> testPairs = setupPairData(keys.subList(0, testSize*3), testSize);
        for (Triplet<String, String, Integer> pair : testPairs)
        {
            String s1 = pair.getValue0();
            String s2 = pair.getValue1();

            List<String> l1 = authorPostContentMap.get(s1);
            List<String> l2 = authorPostContentMap.get(s2);

            postData.put("f,"+s1, l1.get(0));
            for (int i = 1; i < l1.size()-1; i++)
                postData.put(i+","+s1, l1.get(i));
            for (int i = 1; i < l2.size()-1; i++)
                postData.put(i+","+s2, l2.get(i));
            postData.put("l,"+s2, l2.get(l2.size()-1));
        }

        // get imposters
        generateImpostors(authorPostContentMap, postData, keys);

        // generate the vectors for the problem set and imposters
        Map<String, Vector> vectors = NGramVectors.run(postData,TokenType.NGRAMS);

        int featureCount = vectors.get(impostors.get(0)).size();

        // set up randoms
        boolean[][] randFeatures = new boolean[Parameters.K][featureCount];
        List<Integer>[] randImpostors = new List[Parameters.K];
        for (int i = 0; i < Parameters.K; i++)
        {
            randFeatures[i] = ImpostorUtilities.randomFeatures(featureCount, Parameters.FEATURE_SUBSET_PERCENT);
            randImpostors[i] = ImpostorUtilities.randomIndices(impostors.size(), Parameters.IMPOSTERS_PER_IT);
        }

        System.out.println("Testing Pairs");
        List<Triplet<Double, Double, Integer>> results = new ArrayList<>();

        int iteration = 0;
        for (Triplet<String, String, Integer> pair : testPairs)
        {
            String a1 = "f,"+pair.getValue0();
            String a2 = "l,"+pair.getValue1();

            Vector v1 = vectors.get(a1);
            Vector v2 = vectors.get(a2);

            List<String> xImpostors = generateMistakenImpostors(pair.getValue0(), impostors, authorPostContentMap);
            List<String> yImpostors = generateMistakenImpostors(pair.getValue1(), impostors, authorPostContentMap);

            double similarity = 0;
            for (int i = 0; i < Parameters.K; i++)
            {
                List<String> xImpSubset = new ArrayList<>(Parameters.IMPOSTERS_PER_IT);
                List<String> yImpSubset = new ArrayList<>(Parameters.IMPOSTERS_PER_IT);

                for (int index : randImpostors[i])
                {
                    xImpSubset.add(xImpostors.get(index));
                    yImpSubset.add(yImpostors.get(index));
                }
                similarity += ImpostorSimilarity.similarityIteration(v1, v2, xImpSubset, yImpSubset, randFeatures[i], vectors, a1, a2);
            }
            similarity = (similarity/2.)/Parameters.K;
            double minmaxSim = 1-ImpostorUtilities.minMaxSimilarity(v1,v2);

            results.add(new Triplet<>(1 - similarity, minmaxSim, pair.getValue2()));

            Utilities.printPercentProgress(iteration++, testPairs.size());
        }

        List<String> outputLines = pairResults(results, TokenType.IMPOSTOR.toString());
        exportResultsToFile(outputLines, output);
    }

//    protected void cluster(String output, boolean lsh)
//    {
//        if (MISTAKEN_IMPOSTORS == 0)
//        {
//            Map<String, Integer> labels	= new HashMap<>();
//            List<String> clusterPosts = new ArrayList<>();
//            impostors = new ArrayList<>();
//
//            ImpostorCluster cluster = new ImpostorCluster(setupClusterData(labels, clusterPosts),
//                    clusterPosts, impostors);
//            cluster.cluster(labels, output, "impostor", lsh);
//        }
//        else
//            clusterMistakenImpostors(output, lsh);
//    }

    private void clusterMistakenImpostors(String output, boolean lsh)
    {
        //		System.out.println("running mistaken impostors : " + MISTAKEN_IMPOSTORS);
        //		Map<String, Integer> labels	= new HashMap<>();
        //		List<String> clusterPosts = new ArrayList<>();
        //		impostors = new ArrayList<>();
        //
        //		ImpostorCluster cluster = new ImpostorCluster(setupClusterData(labels, clusterPosts),
        //				clusterPosts, impostors);
        //
        //		Map<String, List<String>> bestImpostersMap = getBestImpostors(clusterPosts.toArray(new String[clusterPosts.size()]));
        //		cluster.cluster(labels, output, lsh, bestImpostersMap);
    }


//    private Map<String, List<String>> getBestImpostors(String[] posts, Map<String, Vector> vectorMap)
//    {
//        Map<String, List<String>> bestImpostersMap = ImpostorUtilities.findTopImposters(impostors, vectorMap, posts);
//
//        // This is for testing the negative effects of mistaken impostors
//        if (MISTAKEN_IMPOSTORS > 0)
//        {
//            generateMistakenImpostors(posts, bestImpostersMap);
//        }
//        return bestImpostersMap;
//    }

    private List<String> generateMistakenImpostors(String x, List<String> allImpostors, Map<String, List<String>> authorPostContentMap)
    {
    List<String> xImpostors = new ArrayList<>(allImpostors);
        List<Integer> mistakenImpsIndices, replaceImpsIndices;
        List<String> xPosts;
        int postIndex, impostorIndex;
        // insert self impostors
        xPosts				= authorPostContentMap.get(x.replaceFirst("\\d,", ""));
        // select random posts to serve as mistaken impostors
        mistakenImpsIndices = ImpostorUtilities.randomIndices(xPosts.size()-2, MISTAKEN_IMPOSTORS);
        // replace random impostors at these indices
        replaceImpsIndices  =  ImpostorUtilities.randomIndices(xImpostors.size(), MISTAKEN_IMPOSTORS);
        for (int i = 0; i < MISTAKEN_IMPOSTORS; i++)
        {
            // we want to exclude the first and last posts since we are using them ( (index range-2) + 1)
            postIndex = mistakenImpsIndices.get(i)+1;
            impostorIndex = replaceImpsIndices.get(i);
            xImpostors.set(impostorIndex, postIndex+","+xPosts.get(postIndex));
        }
        return xImpostors;
    }

    protected Map<String, String> setupClusterData(Map<String, Integer> labels, List<String> clusterPosts)
    {
        double uniquePairs = 0;
        double totalPosts = 0;
        boolean allImposters = true;
        RandomDataGenerator zipf = new  RandomDataGenerator();

        Map<String, List<String>> authorPostContentMap = readInData(DATA_LOCATION, SNIPPET_LENGTH, NUM_AUTHORS, false);

        System.out.print("Randomly choosing test data and imposters...");
        Map<String,String> posts = new HashMap<>();
        List<String> keys = new ArrayList<>(authorPostContentMap.keySet());
        String key;
        int postNumber, postCount, index;

        List<String> authorPosts;
        // randomly choose authors to attempt to cluster
        for (int i = 0; i < testSize; i++)
        {
            postCount = 0;
            key = keys.get(rand.nextInt(keys.size()));
            authorPosts = authorPostContentMap.get(key);

            // add only first and last post by author
            clusterPosts.add(1 + "," + key);
            posts.put(1 + "," + key, authorPosts.get(0));
            clusterPosts.add(0 + "," + key);
            posts.put(0 + "," + key, authorPosts.get(authorPosts.size()-1));
            // add author label
            labels.put(1 + "," + key, i);
            labels.put(0 + "," + key, i);

            if (ZIFF_POSTS)
            {
                // remove the posts we've already used
                authorPosts.remove(authorPosts.size()-1);
                authorPosts.remove(0);
                // add random number of other posts by author drawn from zipf
                postCount = zipf.nextZipf(1000, 2)+1;
                // use all other posts by this author
                if (postCount > authorPosts.size())
                    postCount = authorPosts.size();

                for (int j = 2; j < postCount+2; j++)
                {
                    index = rand.nextInt(authorPosts.size());
                    clusterPosts.add(j + "," + key);
                    posts.put(j + "," + key, authorPosts.get(index));
                    authorPosts.remove(index);
                    labels.put(j + "," + key, i);
                }
            }

            keys.remove(key);
            uniquePairs += ArithmeticUtils.binomialCoefficientDouble(postCount+2, 2);
            totalPosts += 2+postCount;
        }

        List<String> imposterPosts;
        if (allImposters)
        {
            // use all the other posts as imposters
            for (String imposter : keys)
            {
                imposterPosts = authorPostContentMap.get(imposter);
                for (int i = 0; i < imposterPosts.size(); i++)
                {
                    impostors.add(i+","+imposter);
                    posts.put(i+","+imposter, imposterPosts.get(i));
                }
            }
        }
        else
        {
            generateImpostors(authorPostContentMap, posts, keys);

        }
        System.out.println(" Done");
        System.out.println("Found " + impostors.size() + " imposters.");
        System.out.println(totalPosts + " total posts.");
        System.out.println(uniquePairs + " Unique same author pairs.");
        return posts;
    }

    private void generateImpostors(Map<String, List<String>> authorPostContentMap, Map<String, String> posts, List<String> keys) {
        String key;
        List<String> imposterPosts;
        int postNumber;// randomly choose imposters
        for (int i = 0; i < TOTAL_IMPOSTERS; i++)
        {
            // choose random author from remaining authors
            key = keys.get(rand.nextInt(keys.size()));
            imposterPosts = authorPostContentMap.get(key);
            // choose a random post from impostor
            postNumber = rand.nextInt(imposterPosts.size());
            impostors.add(i+","+key);
            posts.put(i+","+key, imposterPosts.get(postNumber));

            keys.remove(key);
        }
    }
}