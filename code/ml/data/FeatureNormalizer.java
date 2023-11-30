package ml.data;

import java.util.ArrayList;

/**
 * Class that preprocess data
 * 
 * @author Collins Kariuki
 *
 */

public class FeatureNormalizer implements DataPreprocessor {
    // array lists that will hold the means and standard deviations of the features
    ArrayList<Double> means = new ArrayList<Double>();
    ArrayList<Double> standardDeviations = new ArrayList<Double>();

    /**
     * Calculate the mean of the data
     * 
     * @param data
     * @return the mean
     */
    private static double mean(ArrayList<Double> data) {
        double sum = 0.0;
        for (double num : data) {
            sum += num;
        }
        return sum / data.size();
    }

    /**
     * Calculate the standard deviation of the data
     * 
     * @param data
     * @return the standard deviation
     */
    private static double standardDeviation(ArrayList<Double> data) {
        double mean = mean(data);
        double sumSquaredDiff = 0.0;
        for (double num : data) {
            sumSquaredDiff += Math.pow(num - mean, 2);
        }
        return Math.sqrt(sumSquaredDiff / data.size());
    }

    /**
     * Extract the feature values from the examples
     * 
     * @param examples
     * @param featureIndex
     * @return the feature values
     */
    private ArrayList<Double> extractFeatureValues(ArrayList<Example> examples, int featureIndex) {
        // an array list that will hold the feature values
        ArrayList<Double> featureValues = new ArrayList<Double>();
        for (Example e : examples) {
            // add the feature value for that particular example to the array list
            featureValues.add(e.getFeature(featureIndex));
        }
        return featureValues;
    }

    @Override
    public void preprocessTrain(DataSet train) {
        // get the examples from the training data
        ArrayList<Example> examples = train.getData();
        // get the number of features in the training data
        int featureCount = train.getAllFeatureIndices().size();

        // loop through the features
        for (int i = 0; i < featureCount; i++) {
            // extract the feature values
            ArrayList<Double> featureValues = extractFeatureValues(examples, i);
            // calculate the mean of the feature values
            double mean = mean(featureValues);
            // add the mean to the array list
            means.add(mean);

            // center data to 0
            for (Example e : examples) {
                e.setFeature(i, e.getFeature(i) - mean);
            }

            // normalize to have standard deviation of 1
            double standardDeviation = standardDeviation(featureValues);
            standardDeviations.add(standardDeviation);

            for (Example e : examples) {
                e.setFeature(i, e.getFeature(i) / standardDeviation);
            }
        }
    }

    @Override
    public void preprocessTest(DataSet test) {
        // get the examples from the test data
        ArrayList<Example> examples = test.getData();
        // get the number of features in the test data
        int featureCount = test.getAllFeatureIndices().size();

        // check if the number of features in the test data is the same as the number of
        // features in the training data
        if (featureCount != means.size()) {
            throw new IllegalArgumentException("Test set has different number of features than train set");
        }

        // loop through the features
        // we use the means and standard deviations from the training data and apply it
        // to the test data following the mantra that what we do to the training data
        // we must do to the test data
        for (int i = 0; i < featureCount; i++) {
            // get the mean and standard deviation for that feature
            double mean = means.get(i);
            // get the standard deviation for that feature
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