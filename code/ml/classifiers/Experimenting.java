package ml.classifiers;

import java.util.ArrayList;

import ml.data.CrossValidationSet;
import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;
import ml.data.ExampleNormalizer;
import ml.data.FeatureNormalizer;

public class Experimenting {
	public Experimenting() {
	}

	/**
	 * Main function to test gradient descent algorithm
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		RandomClassifier cl = new RandomClassifier();
		DataSet data = new DataSet("data/diabetesDecimalLabel.csv", 0); // 0 for csv file
		CrossValidationSet crossValidation = new CrossValidationSet(data, 10, true);

		// run for the number of splits
		ArrayList<Double> splitAvgs = new ArrayList<Double>();
		for (int i = 0; i < crossValidation.getNumSplits(); i++) {
			DataSetSplit dataSplit = crossValidation.getValidationSet(i);

			// preprocess data
			FeatureNormalizer featureNormalizer = new FeatureNormalizer();
			ExampleNormalizer exampleNormalizer = new ExampleNormalizer();
			featureNormalizer.preprocessTrain(dataSplit.getTrain());
			exampleNormalizer.preprocessTrain(dataSplit.getTrain());

			cl.train(dataSplit.getTrain());
			double allAccuracy = 0.0;

			// runs classifier 100 times to find accuracy
			for (int j = 0; j < 100; j++) {
				double correctCount = 0.0;
				for (Example e : dataSplit.getTrain().getData()) {
					double prediction = cl.classify(e);
					if (prediction == e.getLabel()) {
						correctCount += 1;
					}
				}
				double currentAccuracy = correctCount / dataSplit.getTrain().getData().size();
				allAccuracy += currentAccuracy;
			}
			double avg = allAccuracy / 100;
			splitAvgs.add(avg); // will hold the avg accuragy from thr ith split
		}
		System.out.println(splitAvgs);

		double sumAvgs = 0.0;
		for (int i = 0; i < splitAvgs.size(); i++) {
			sumAvgs += splitAvgs.get(i);
		}

		System.out.println(sumAvgs / crossValidation.getNumSplits());

	}
}
