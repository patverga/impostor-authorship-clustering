package co.pemma.authorVerification.cluster;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instance;
import weka.core.Instances;


public class SVM 
{
	SMO classifier;
	Instances trainVectors;
	Instances testVectors;

	public SVM(int vectorSize)
	{
		this.classifier = new SMO();
		try 
		{
			this.classifier.setOptions(weka.core.Utils.splitOptions(""
//							+ "-K \"weka.classifiers.functions.supportVector.PolyKernel -E 3.0\""
					));
		} 
		catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		System.out.println("KERNEL : " + classifier.getKernel());
		//		classifier.setOptions(new String[] { "-R" });
	}

	public void batchTrain(Instances train)
	{
		this.trainVectors = train;
		try 
		{
			classifier.buildClassifier(trainVectors);

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
			Evaluation eTest = new Evaluation(trainVectors);
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
	private double classifyVector(Instance i) 
	{
		i.setDataset(trainVectors);
		
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

	public void save()
	{

	}

	public void load()
	{	
	}	
}
