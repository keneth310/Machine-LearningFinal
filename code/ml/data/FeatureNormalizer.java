package ml.data;

import java.util.ArrayList;

/**
 * Class that preprocess data
 * 
 * @author Collins Kariuki
 *
 */

public class FeatureNormalizer implements DataPreprocessor{
    /**
	 * Preprocess the training data
	 * 
	 * @param train
	 */
    ArrayList<Double> means = new ArrayList<Double>();
    ArrayList<Double> standardDeviations = new ArrayList<Double>();

    private static double mean(ArrayList<Double> data) {
        double sum = 0.0;
        for (double num : data) {
            sum += num;
        }
        return sum / data.size();
    }

    private static double standardDeviation(ArrayList<Double> data) {
        double mean = mean(data);
        double sumSquaredDiff = 0.0;
        for (double num : data) {
            sumSquaredDiff += Math.pow(num - mean, 2);
        }
    return Math.sqrt(sumSquaredDiff / data.size());
    }

    private ArrayList<Double> extractFeatureValues(ArrayList<Example> examples, int featureIndex) {
        ArrayList<Double> featureValues = new ArrayList<Double>();
        for (Example e : examples) {
            featureValues.add(e.getFeature(featureIndex));
        }
        return featureValues;
    }

    @Override
    public void preprocessTrain(DataSet train) {
        ArrayList<Example> examples = train.getData();
        int featureCount = train.getAllFeatureIndices().size();

        for (int i = 0; i < featureCount; i++) {
            ArrayList<Double> featureValues = extractFeatureValues(examples, i);
            double mean = mean(featureValues);
            means.add(mean);

            // Center data to 0
            for (Example e : examples) {
                e.setFeature(i, e.getFeature(i) - mean);
            }

            // Normalize to have standard deviation of 1
            double standardDeviation = standardDeviation(featureValues);
            standardDeviations.add(standardDeviation);

            for (Example e : examples) {
                e.setFeature(i, e.getFeature(i) / standardDeviation);
            }
        }
    }
	
    @Override
    public void preprocessTest(DataSet test) {
        ArrayList<Example> examples = test.getData();
        int featureCount = test.getAllFeatureIndices().size();

        if (featureCount != means.size()) {
            throw new IllegalArgumentException("Test set has different number of features than train set");
        }

        for (int i = 0; i < featureCount; i++) {
            double mean = means.get(i);
            double standardDeviation = standardDeviations.get(i);

            // Center data to 0
            for (Example e : examples) {
                e.setFeature(i, e.getFeature(i) - mean);
            }

            // Normalize to have standard deviation of 1
            for (Example e : examples) {
                e.setFeature(i, e.getFeature(i) / standardDeviation);
            }
        }
    }
}