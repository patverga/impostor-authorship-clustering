package co.pemma.authorVerification.experiments;

import co.pemma.authorVerification.*;
import co.pemma.authorVerification.NGramVectors.TokenType;
import co.pemma.authorVerification.Utilities;
import org.apache.mahout.common.distance.CosineDistanceMeasure;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Vector;
import org.javatuples.Pair;
import org.javatuples.Triplet;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


public class BackPageExperiment extends Experiment
{
    Map<String, Integer> postLabelMap = new HashMap<>();
    Map<Integer, List<String>> labelPostListMap = new HashMap<>();
    List<String> allImposters;
    Map<String, String> impostorData;
    Map<String, String> postData;
    String city = null;
    String phone = "[2-9]{3}\\D[1-9]{3}\\D[1-9]{4}";

    public BackPageExperiment()
    {
        super();
        DATA_LOCATION = "data/backpage";
        this.allImposters = new ArrayList<>();
        this.impostorData 	= new ConcurrentHashMap<>();
        this.postData		= new ConcurrentHashMap<>();
        this.postLabelMap = new HashMap<>();
        this.labelPostListMap = new HashMap<>();
        readInDataWithLabels(postLabelMap, labelPostListMap);
        readInImposters();
        //				allImposters = allImposters.subList(0, TOTAL_IMPOSTERS);
    }

    public BackPageExperiment(String c)
    {
        this();
        if (c!= null)
            this.city = c.toLowerCase();
        else this.city = null;
    }

    private void readInImposters()
    {
        // use the non-test blogs as imposters
        System.out.println("Reading in imposters...");
        final File directory = new File(DATA_LOCATION + "/imposters");
//        final File directory = new File(DATA_LOCATION + "/other_impostors");

        ExecutorService executor = Executors.newFixedThreadPool(Utilities.MAX_THREADS);
        File[] allFiles = directory.listFiles();
        File[] subFiles;

        int fileCount = allFiles.length;
        int division = fileCount / Utilities.MAX_THREADS;
        int index, start, end;

        // iterate over each pair subdirectory
        for (int i = 0; i < Utilities.MAX_THREADS; i++)
        {
            index = 0;
            start = (i*division);
            if (i < Utilities.MAX_THREADS - 1)
                end = ((i+1)*division) -1;
            else
                end = fileCount;
            subFiles = new File[end-start];

            for (int j =start; j < end; j++)
            {
                subFiles[index] = allFiles[j];
                allImposters.add(subFiles[index++].getName());
            }
            executor.submit(new ReadInDataThread(impostorData, subFiles));
        }
        Utilities.waitForThreads(executor);
        System.out.println("Read in " + allImposters.size() + " total imposters.");
    }

    /**
     * imposters list starts with containing all files, test and train start empty
     * each problem iteration adds a random pair to test and train and removes them
     * from the imposter set (might be better to choose pairs first and remove from imps)
     */

    public void pairs(String outputLocation, TokenType feature)
    {
        reset();
        Map<String, String> vectorData = new ConcurrentHashMap<>();

        List<Pair<String, String>> testPairs = new ArrayList<>();
        int[] labels = choosePairs(testPairs, vectorData, new ArrayList<>(labelPostListMap.keySet()), testSize);

        // get imposters
        impostors = generateImposters(vectorData);
        System.out.println("Randomly selected " + impostors.size() + " imposters.");
        Map<String, Vector> vectorMap;
        DistanceMeasure cosine = null;
        switch(feature)
        {
            case IMPOSTOR:
                vectorMap = NGramVectors.run(vectorData, TokenType.NGRAMS);
                break;
            case STYLO:
                cosine = new CosineDistanceMeasure();
                vectorMap =  WritePrintsVectors.jstyloVectors(vectorData);
                break;
            default:
                vectorMap = NGramVectors.run(vectorData, feature);
                break;
        }

        List<Triplet<Double, Double, Integer>> results = testPairs(testPairs, labels, vectorMap, feature, cosine);

        List<String> outputLines = pairResults(results, feature.toString());
        exportResultsToFile(outputLines, outputLocation);
    }

    private List<Triplet<Double, Double, Integer>> testPairs(List<Pair<String, String>> testPairs,
                                                  int[] labels, Map<String, Vector> vectorMap, TokenType feature, DistanceMeasure cosine)
    {
        int featureCount = vectorMap.get(impostors.get(0)).size();

        // set up randoms
        boolean[][] randFeatures = new boolean[Parameters.K][featureCount];
        List<Integer>[] randImpostors = new List[Parameters.K];
        for (int i = 0; i < Parameters.K; i++)
        {
            randFeatures[i] = ImpostorUtilities.randomFeatures(featureCount, Parameters.FEATURE_SUBSET_PERCENT);
            randImpostors[i] = ImpostorUtilities.randomIndices(impostors.size(), Parameters.IMPOSTERS_PER_IT);
        }

        List<Triplet<Double, Double, Integer>> results = new ArrayList<>();

        // test each pair
        System.out.println("Testing pairs...");
        double similarity;
        for (int i = 0; i < testPairs.size(); i++)
        {
            Pair<String, String> pair = testPairs.get(i);
            String a1 = pair.getValue0();
            String a2 = pair.getValue1();

            Vector v1 = vectorMap.get(a1);
            Vector v2 = vectorMap.get(a2);

            similarity = 0;
            switch(feature)
            {
                case IMPOSTOR:

                    for (int k = 0; k < Parameters.K; k++)
                    {
                        List<String> imps = new ArrayList<>(Parameters.IMPOSTERS_PER_IT);
                        for (int index : randImpostors[k])
                            imps.add(impostors.get(index));
                        similarity += ImpostorSimilarity.similarityIteration(v1, v2, imps, imps, randFeatures[k], vectorMap, a1, a2);
                    }
                    similarity = (similarity/2.)/Parameters.K;
                    break;

                case STYLO:
                    if (v1 != null && v2 != null)
                        similarity = 1.- cosine.distance(v1, v2);
                    break;
                default:
                    similarity = ImpostorUtilities.minMaxSimilarity(v1, v2);
                    break;
            }
            double minmaxSim = 1-ImpostorUtilities.minMaxSimilarity(v1,v2);

            results.add(new Triplet<>(1.-similarity, minmaxSim, labels[i]));
//			updateResultCounts(similarity, labels[i], sameAuthor, diffAuthor);
            Utilities.printPercentProgress(i, testPairs.size());
        }
        return results;
    }

    @Override
    public void learnPairs(String output, String classifierType, boolean ngrams, boolean tfidf, boolean writeprints)
    {
        Classifier classifier = new Classifier(classifierType);
        reset();
        Map<String, String> vectorData = new ConcurrentHashMap<>();
        List<Integer> keys = new ArrayList<>(labelPostListMap.keySet());

        // set up test data
        List<Pair<String, String>> testPairs = new ArrayList<>();
        int[] testLabels = choosePairs(testPairs, vectorData, keys, testSize);

        // set up training data
        List<Pair<String, String>> trainPairs = new ArrayList<>();
        int[] trainLabels = choosePairs(trainPairs, vectorData, keys, trainSize);

        // get imposters
        impostors = generateImposters(vectorData);

        // generate the vectors for the problem set and imposters
        Map<String, Vector> ngramVectors = NGramVectors.run(vectorData);
        Map<String, Vector> tfidfVectors = null;
        Map<String, Vector> writeprintsVectors = null;

        // train model
        ArrayList<Attribute> features = trainPairs(trainPairs, trainLabels, ngramVectors, tfidfVectors, writeprintsVectors, ngrams, tfidf, writeprints, vectorData, classifier);

        // test each pair
        System.out.println("Testing pairs...");
        Instances testSet = new Instances("test", features, testPairs.size());
        testSet.setClassIndex(testSet.numAttributes()-1);
        List<Triplet<Double, Double, Integer>> results = new ArrayList<>();
        for (int i = 0; i < testPairs.size(); i++)
        {
            Instance in = similarityVector(testPairs.get(i).getValue0(), testPairs.get(i).getValue1(), features, ngrams, ngramVectors, tfidfVectors, writeprintsVectors);
            if (in != null)
            {
                in.setDataset(testSet);
                in.setClassValue(""+testLabels[i]);
                testSet.add(in);
                results.add( new Triplet<>(1-classifier.classifyVector(in), 1.,testLabels[i]));
            }
            Utilities.printPercentProgress(i, testPairs.size());
        }
        classifier.batchTest(testSet);

        String learningParams = "\t"+classifierType+"\t"+ngrams+"\t"+tfidf+"\t"+writeprints+"\t"+trainSize;

        List<String> outputLines = pairResults(results, learningParams);
        exportResultsToFile(outputLines, output);
    }

    @Override
    protected void mistakenImpostors(String output, int mistakenImpostors, int divisions) {

    }

    public List<String> generateImposters(Map<String, String> vectorData)
    {
        List<String> imposterSubset = new ArrayList<String>();
        //		List<Integer> useImpostors = ImpostorUtilities.randomIndices(allImposters.size(), TOTAL_IMPOSTERS);
        List<Integer> useImpostors = ImpostorUtilities.randomIndices(allImposters.size(), allImposters.size());

        for (int i = 0; i < allImposters.size(); i++ )
        {
            if (useImpostors.contains(i))
            {
                imposterSubset.add(allImposters.get(i));
                vectorData.put(allImposters.get(i), impostorData.get(allImposters.get(i)));
            }
        }
        return imposterSubset;
    }

    public List<String> readInDataWithLabels(Map<String, Integer> postLabelMap, Map<Integer, List<String>> labelPostListMap)
    {
        System.out.println("Reading in data");
        final File directory = new File(DATA_LOCATION+"/raw_txt/0/");

        // read in label file

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(DATA_LOCATION+"/labels"))))
        {
            String line, id;
            int label;
            String[] parts;
            while ((line = reader.readLine()) != null)
            {
                parts = line.split(",");
                id = parts[0];
                if (parts.length < 2)
                    System.out.println(line);
                label = Integer.parseInt(parts[1]);
                postLabelMap.put(id, label);
            }
        }
        catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        // read in the actual post data
        List<String> postKeys = new ArrayList<String> ();
        ExecutorService executor = Executors.newFixedThreadPool(Utilities.MAX_THREADS);
        List<String> posts;
        String key;
        int label;
        File[] allFiles = directory.listFiles();
        assert allFiles != null;
        int division = allFiles.length / Utilities.MAX_THREADS;
        int threadIndex = 1, fileIndex = 0;
        File[] subFiles = new File[division];
        File file;
        for (int i = 0; i < allFiles.length; i++)
        {
            file = allFiles[i];
            subFiles[fileIndex++] = file;
            key = file.getName();
            if (postLabelMap.containsKey(key))
            {
                label = postLabelMap.get(key);
                postKeys.add(key);
                if (labelPostListMap.containsKey(label))
                {
                    posts = labelPostListMap.get(label);
                    posts.add(key);
                }
                else
                {
                    posts = new ArrayList<>();
                    posts.add(key);
                    labelPostListMap.put(label, posts);
                }
            }
            if (fileIndex >= subFiles.length)
            {
                executor.submit(new ReadInDataThread(postData, subFiles));
                fileIndex = 0;
                if (++threadIndex == Utilities.MAX_THREADS)
                    subFiles = new File[allFiles.length - (i+1)];
                else
                    subFiles = new File[division];
            }
        }
        Utilities.waitForThreads(executor);
        System.out.println("Found " + postData.size() + " posts.");

        return postKeys;
    }

    public class ReadInDataThread implements Runnable
    {
        private File[] posts;
        private Map<String, String> dataMap;

        ReadInDataThread(Map<String, String> dataMap, File... posts)
        {
            this.posts = posts;
            this.dataMap = dataMap;
        }

        @Override
        public void run()
        {
            for (int i =0; i < posts.length; i++)
            {
                File post = posts[i];
                if (!post.isDirectory())
                {
                    try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(post))))
                    {
                        //						System.out.println();
                        //						System.out.println(post);
                        String text = "", line;
                        while ((line = reader.readLine()) != null)
                        {
                            //							System.out.println(line);
                            text += line.replaceAll("\\d", "");
                            //							System.out.println();
                            //							System.out.println(line.replaceAll("\\d", ""));
                            //							System.out.println("\n");
                        }
                        dataMap.put(post.getName(), text);
                    }
                    catch (Exception e) {
                        // TODO Auto-generated catch block
                        e.printStackTrace();
                    }
                }
            }
        }
    }

    /**
     * @param labels will map post keys to their true label
     * @param clusterPosts will be filled with the post keys that we want to cluster
     * @return map from postkeys to post data
     */
    @Override
    protected Map<String, String> setupClusterData(Map<String, Integer> labels,	List<String> clusterPosts)
    {
        // add the posts that we want to cluster
        Map<String, String> posts = new HashMap<>();
        List<Integer> authors = new ArrayList<>(labelPostListMap.keySet());
        // randomly choose test and train authors
        if (city != null)
            selectRandomAuthorsSingleCity(labels, clusterPosts, posts, authors, testSize, 2, city);
        else if (threshold < 1.0)
            selectRandomAuthorsThreshold(labels, clusterPosts, posts, authors, testSize, 2, threshold, NGramVectors.run(postData));
        else if (merge < 1.0)
            selectRandomAuthorsWithMerge(labels, clusterPosts, posts, authors, testSize, 2, threshold, NGramVectors.run(postData));
        else
            selectRandomAuthors(labels, clusterPosts, posts, authors, testSize, 2);

        // add the impostors
        impostors = generateImposters(posts);
        System.out.println("Selected " + clusterPosts.size() + " posts to cluster.");

        return posts;
    }

    private List<Integer> selectRandomAuthors(Map<String, Integer> labels, List<String> clusterPosts,
                                              Map<String, String> posts, List<Integer> authors, int count, int minPosts )
    {
        List<Integer> authorList = new ArrayList<>();
        int author;
        int index;
        while (authorList.size() < count)
        {
            index = rand.nextInt(authors.size());
            author = authors.get(index);
            if (labelPostListMap.get(author).size() >= minPosts)
            {
                for (String post : labelPostListMap.get(author))
                {
                    posts.put(post, postData.get(post));
                    labels.put(post, author);
                    clusterPosts.add(post);
                }
                authorList.add(author);
            }
            authors.remove(index);
        }
        return authorList;
    }

    private List<Integer> selectRandomAuthorsSingleCity(Map<String, Integer> labels, List<String> clusterPosts,
                                                        Map<String, String> posts, List<Integer> authors, int count, int minPosts, String city)
    {
        List<Integer> authorList = new ArrayList<>();
        String c;
        int author;
        int index;
        int cityCount;
        while (authorList.size() < count)
        {
            index = rand.nextInt(authors.size());
            author = authors.get(index);
            cityCount = 0;
            for (String post : labelPostListMap.get(author))
            {
                c = post.split("_")[0];
                if (c.equals(city))
                    cityCount++;
            }
            if (cityCount >= minPosts)
            {
                for (String post : labelPostListMap.get(author))
                {
                    c = post.split("_")[0];
                    if (c.equals(city))
                    {
                        posts.put(post, postData.get(post));
                        labels.put(post, author);
                        clusterPosts.add(post);
                    }
                }
                authorList.add(author);
            }
            authors.remove(index);
        }
        return authorList;
    }

    private List<Integer> selectRandomAuthorsWithMerge(Map<String, Integer> labels, List<String> clusterPosts, Map<String, String> posts,
                                                       List<Integer> authors, int count, int minPosts, double threshold, Map<String, Vector> vectorMap)
    {
        System.out.println("Selecting random authors - Merging similar ones.");
        List<Integer> authorList = new ArrayList<>();
        List<String> authorPosts;
        int author;
        int label;
        int index;
        double similarity;
        Vector v1, v2;
        while (authorList.size() < count)
        {
            index = rand.nextInt(authors.size());
            author = authors.get(index);
            label = author;
            authorPosts =labelPostListMap.get(author);
            if ( authorPosts.size() > minPosts)
            {
                // make sure this author isnt too similar to any chosen author so far
                for (String post : authorPosts) {
                    v1 = vectorMap.get(post);
                    for (int chosenAuthor : authorList) {
                        for (String post2 : labelPostListMap.get(chosenAuthor)){
                            v2 = vectorMap.get(post2);
                            similarity = ImpostorUtilities.minMaxSimilarity(v1, v2);
                            // if its too similar, merge the new author into the similar old one
                            if (similarity > merge) {
                                System.out.println("MERGE: " + similarity);
                                label = chosenAuthor;
                                count += 1; // because we're still adding author to list, just labeling its post differently
                                break;
                            }
                        }
                        if (label != author)
                            break;
                    }
                    if (label != author)
                        break;
                }
                for (String post : authorPosts)
                {
                    posts.put(post, postData.get(post));
                    labels.put(post, label);
                    clusterPosts.add(post);
                }
                authorList.add(author);
            }
            authors.remove(index);
        }
        return authorList;
    }

    private List<Integer> selectRandomAuthorsThreshold(Map<String, Integer> labels, List<String> clusterPosts,
                                                       Map<String, String> posts, List<Integer> authors, int count, int minPosts, double threshold, Map<String, Vector> vectorMap)
    {
        System.out.println("thresholding");
        List<Integer> authorList = new ArrayList<>();
        ArrayList<String> authorPosts;
        int author;
        int index;
        double simTotal;
        Vector v1, v2;
        while (authorList.size() < count)
        {
            index = rand.nextInt(authors.size());
            author = authors.get(index);
            authorPosts = new ArrayList<>(labelPostListMap.get(author));
            if ( authorPosts.size() > minPosts)
            {
                simTotal = 0;
                for (int i = 0; i < authorPosts.size(); i++)
                {
                    v1 = vectorMap.get(authorPosts.get(i));
                    for (int j = i+1; j < authorPosts.size(); j++)
                    {
                        v2 = vectorMap.get(authorPosts.get(j));
                        simTotal += ImpostorUtilities.minMaxSimilarity(v1, v2);
                    }
                }
                // only use authors who average similarity is below the threshold
                if (simTotal/authorPosts.size() < threshold )
                {
                    for (String post : authorPosts)
                    {
                        posts.put(post, postData.get(post));
                        labels.put(post, author);
                        clusterPosts.add(post);
                    }
                    authorList.add(author);
                }
            }
            authors.remove(index);
        }
        return authorList;
    }

    private int[] choosePairs(List<Pair<String, String>> testPairs, Map<String, String> vectorData, List<Integer> authors, int size)
    {
        int[] labels = new int[size*2];
        List<String> authorList = null;
        String post1, post2;
        int index;
        int otherIndex;
        int author, otherAuthor;

        // randomly select same author and different author pairs to test
        boolean firstTry;
        int i = 0;
        while (testPairs.size() < size*2)
        {
            firstTry = true;
            // select random author until we find one with atleast 2 posts
            while (firstTry || authorList.size() < 2)
            {
                index = rand.nextInt(authors.size());
                author = authors.get(index);
                otherAuthor = author;
                authors.remove(index);
                authorList = labelPostListMap.get(author);
                firstTry = false;
            }
            // select random post by this author
            index = rand.nextInt(authorList.size());
            post1 = authorList.get(index);
            authorList.remove(index);

            // make every other iteration a different author pair
            if (i % 2 == 0)
            {
                firstTry = true;
                while (firstTry || authorList.size() < 2)
                {
                    firstTry = false;
                    otherIndex = rand.nextInt(authors.size());
                    otherAuthor = authors.get(otherIndex);
                    authorList = labelPostListMap.get(otherAuthor);
                    authors.remove(otherIndex);
                }
            }
            labels[i] = i % 2;
            index = rand.nextInt(authorList.size());
            post2 = authorList.get(index);

            vectorData.put(post1, postData.get(post1));
            vectorData.put(post2, postData.get(post2));

            testPairs.add(new Pair<>(post1, post2));
            i++;
        }
        return labels;
    }


    //	public static void main(String[] args)
    //	{
    //
    //		String[] test = new String[]{
    //				"223 asdadsd 234-2224",
    //				"223.234-2224",
    //				"1231231234",
    //				"234-2342345"
    //		};
    //		String phone = "[2-9]{3}.?\\p{Punct}*[1-9]{3}.?\\p{Punct}*[1-9]{4}";
    //
    //		for(String s : test )
    //		{
    //			System.out.println(s.replaceAll("\\d", ""));
    //		}
    //	}
}
