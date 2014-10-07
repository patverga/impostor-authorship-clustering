package co.pemma.authorVerification.cluster;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;

import org.apache.mahout.classifier.sgd.AdaptiveLogisticRegression;
import org.apache.mahout.classifier.sgd.CrossFoldLearner;
import org.apache.mahout.classifier.sgd.L1;
import org.apache.mahout.classifier.sgd.ModelSerializer;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

public class Regression
{
	AdaptiveLogisticRegression reg;
	CrossFoldLearner model;
	String modelPath = "no where im not saving this right now";

	public Regression(int vectorSize)
	{
		this.reg = new AdaptiveLogisticRegression(2, vectorSize, new L1());
	}

	public void batchTrain(List<Vector> trainVectors, int[] labels)
	{
		for (int n = 0; n < 10000; n++)
		{
			for(int i = 0 ; i < trainVectors.size(); i++) 
				reg.train(labels[i], trainVectors.get(i));
		}
		reg.close();
		model = reg.getBest().getPayload().getLearner();
	}

	public void batchTest(List<Vector> testVectors, int[] labels)
	{
		double correct = 0, total = labels.length;
		int expected, classified;
		Vector v;
		for(int i = 0; i < testVectors.size(); i++)
		{
			expected = labels[i];
			v = testVectors.get(i);

			if (classifyVector(v) >= .5)
				classified = 1;
			else
				classified = 0;
			if (classified == expected) 
				correct++;
		}
		System.out.println("accuracy : " + correct / total);
	}

	/**
	 * 
	 * @param v the vector the classify
	 * @return P(v = 1)
	 */
	private double classifyVector(Vector v) 
	{
		Vector probabilities = new DenseVector(2);
		model.classifyFull(probabilities, v);

		System.out.println(probabilities.get(1));
		return probabilities.get(1);
	}

	public void save()
	{
		try 
		{
			ModelSerializer.writeBinary(modelPath, reg.getBest().getPayload().getLearner());
		} 
		catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public void load()
	{		
		try (InputStream in = new FileInputStream(modelPath)) 
		{
			this.model = ModelSerializer.readBinary(in, CrossFoldLearner.class);
		} 
		catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}	
}
