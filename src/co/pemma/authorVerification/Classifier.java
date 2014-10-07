package co.pemma.authorVerification;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;


import org.apache.mahout.math.Vector;

import co.pemma.authorVerification.Utilities;
import edu.stanford.nlp.util.Triple;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;


public class Classifier
{
    AbstractClassifier classifier;
    Instances trainVectors;
    Instances testVectors;
    String type;
    public int size;

    public Classifier(AbstractClassifier classifier)
    {
        this.classifier = classifier;
    }

    public Classifier(String type)
    {
        this.type = type;

        switch(type)
        {
            case "svm":
                this.classifier = new SMO();
                break;
            case "bayes":
                this.classifier = new NaiveBayesMultinomial();
                break;
            default:
                this.classifier = new Logistic();
                break;
        }
    }

    public void batchTrain(Instances train)
    {
        this.setTrainVectors(train);
        this.size = train.firstInstance().numAttributes();
        try
        {
            classifier.buildClassifier(getTrainVectors());

        }
        catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    public void batchTest(Instances test)
    {
        System.out.println("TST SIZE : " + test.size());
        this.testVectors = test;
        try {
            Evaluation eTest = new Evaluation(getTrainVectors());
            eTest.evaluateModel(classifier, testVectors);

            // Print the result à la Weka explorer:
            String strSummary = eTest.toSummaryString();
            System.out.println(strSummary);
        }
        catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    /**
     *
     * @param v the vector the classify
     * @return P(v = 1)
     */
    public double classifyVector(Instance i)
    {
        i.setDataset(getTrainVectors());

        double[] probDistribution = null;
        try
        {
            probDistribution = classifier.distributionForInstance(i);
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return probDistribution[0];
    }

    public <T> void learnClusterModel(Map<String, T> postLabels, List<String> impostors,
                                      List<String> postKeys, Map<String, String> postData, Map<T, List<String>> labelPostListMap)
    {
//		System.out.println("Training " + type + " model...");
//		int trainSize = 1000;
//
//		List<T> labels = new ArrayList(postLabels.keySet());
//		Map<String, Integer> postIndexMap = new HashMap<>();
//		for (int i = 0; i < postKeys.size(); i++)
//			postIndexMap.put(postKeys.get(i), i);
//
//
//		final List<String> classLabels = new ArrayList<String>() {{add("1");add("0");}};
//		//		ArrayList<Attribute> features = new ArrayList<Attribute>() {{add(new Attribute("impostor"));add(new Attribute("ratio"));add(new Attribute("count"));add(new Attribute("class", classLabels));}};
//		ArrayList<Attribute> features = new ArrayList<Attribute>() {{add(new Attribute("impostor"));add(new Attribute("class", classLabels));}};
//
//		Instances trainSet = new Instances("train", features, trainSize);    
//		trainSet.setClassIndex(trainSet.numAttributes()-1);
//
//		// create vectors our of post data
//		Map<String, Vector> vectorMap = NGramVectors.run(postData, false);
//
//		Random rand = new Random();	
//		List<String> authorPosts1, authorPosts2 = null;
//		int index;
//		T label;
//		while(labels.size() > 1)
//		{			
//			// randomly select 2 authors
//			index = rand.nextInt(labels.size());
//			label = postLabels.get(labels.get(index));
//			labels.remove(index);
//			authorPosts1 = labelPostListMap.get(label);
//			index = rand.nextInt(labels.size());
//			label = postLabels.get(labels.get(index));
//			labels.remove(index);
//			authorPosts2 = labelPostListMap.get(label);
//
//			// keep numnber of postive and negative training examples the same by using the smaller author for same author set
//			if (authorPosts1.size() < authorPosts2.size())
//			{			
//				// same author clusters
//				extractTrainingSamples(postIndexMap, vectorMap, impostors, trainSet, authorPosts1, authorPosts1, 1);
//				// different author clusters
//				extractTrainingSamples(postIndexMap, vectorMap, impostors, trainSet, authorPosts1, authorPosts2, 0);
//			}
//			else
//			{
//				// same author clusters
//				extractTrainingSamples(postIndexMap, vectorMap, impostors, trainSet, authorPosts2, authorPosts2, 1);
//				// different author clusters
//				extractTrainingSamples(postIndexMap, vectorMap, impostors, trainSet, authorPosts2, authorPosts1, 0);
//			}
//			Utilities.printPercentProgress(postLabels.size() - labels.size(), postLabels.size());
//		}
//		batchTrain(trainSet);		
    }

    private void extractTrainingSamples(Map<String, Integer> postIndexMap, Map<String, Vector> vectorMap,
                                        List<String> impostors, Instances trainSet, List<String> authorPosts1, List<String> authorPosts2, int classLabel)
    {
        ImpostorSimilarity impostorSim;
        int size = authorPosts1.size()/2;
        double[] xLastSim = new double[size];
        double yLastSim = 0, similarity, thisSim, xySim;
        int xCount = 0, yCount = 0;
        String x, y;
        Instance in;
        for (int i = 0; i < size; i ++)
            xLastSim[i] = 0;

        //		for (int i = 0; i < size; i++)
        //		{
        //			xCount++;
        //			x = authorPosts1.get(i);
        //			for (int j = size; j < size*2; j++)
        //			{
        //				y = authorPosts2.get(j);
        //
        //
        //				xySim = impostorSim.run();
        //				thisSim = (xySim + (xLastSim[j-size]*(xCount-1)));
        //				similarity = (thisSim + (yLastSim * yCount)) / (xCount + yCount);
        //
        //				xLastSim[j-size] = similarity;
        //				yLastSim = similarity;
        //				yCount++;
        //
        //				// set training vector
        ////				in = new DenseInstance(4);
        ////				in.setDataset(trainSet);
        ////				in.setClassValue(classLabel);
        ////				in.setValue(0, similarity);
        ////				in.setValue(1, Math.min(xCount,yCount)/Math.max(xCount, yCount));
        ////				in.setValue(2, xCount+yCount);
        //				in = new DenseInstance(2);
        //				in.setDataset(trainSet);
        //				in.setClassValue(classLabel);
        //				in.setValue(0, xySim);
        //				trainSet.add(in);
        //			}
        //		}
        x = authorPosts1.get(0);

        y = authorPosts2.get(1);
//		impostorSim = new ImpostorSimilarity(x, y, vectorMap, Parameters.K);

        in = new DenseInstance(2);
        in.setDataset(trainSet);
        in.setClassValue(classLabel);
//		in.setValue(0, impostorSim.run(impostors));
        trainSet.add(in);
    }

    public void save()
    {

    }

    public void load()
    {
    }

    public ArrayList<Attribute> trainCluster(List<Triple<Double, Double, Integer>> trainSet)
    {
        // train the classifier model
        System.out.println("Training regression model...");
        ArrayList<Attribute> features = new ArrayList<Attribute>();
        features.add(new Attribute("impostor"));
//		features.add(new Attribute("ratio"));

        List<String> classLabels = new ArrayList<String>();
        classLabels.add("1");
        classLabels.add("0");

        features.add(new Attribute("class", classLabels));

        Instances trainInstances = new Instances("train", features, trainSet.size());
        trainInstances.setClassIndex(trainInstances.numAttributes()-1);
        Triple<Double, Double, Integer> sample;
        for (int i = 0; i < trainSet.size(); i++)
        {
            sample = trainSet.get(i);
            Instance in = new DenseInstance(features.size());

            in.setDataset(trainInstances);
            in.setValue(features.get(0), sample.first());
//			in.setValue(features.get(1), sample.second());
            in.setClassValue(sample.third());
            trainInstances.add(in);

            Utilities.printPercentProgress(i, trainSet.size());
        }
        batchTrain(trainInstances);

        return features;
    }

    public Instances getTrainVectors() {
        return trainVectors;
    }

    public void setTrainVectors(Instances trainVectors) {
        this.trainVectors = trainVectors;
    }
}
