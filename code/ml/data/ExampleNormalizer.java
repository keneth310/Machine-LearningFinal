package ml.data;

import java.util.ArrayList;
import java.util.Set;

public class ExampleNormalizer implements DataPreprocessor {

    @Override
    public void preprocessTrain(DataSet train) {
        ArrayList<Example> examples = train.getData();
        for (Example e : examples) {
            Set<Integer> featureIndices = e.getFeatureSet();

            double normalization = 0.0;
            // for each feature, add the square of the value to the normalization
            for (int index : featureIndices) {
                double featureValue = e.getFeature(index);
                normalization += Math.pow(featureValue, 2);
            }
            normalization = Math.sqrt(normalization);

            for (int index : featureIndices) {
                double updatedValue = e.getFeature(index) / normalization;
                e.setFeature(index, updatedValue);
            }
        }
    }

    @Override
    public void preprocessTest(DataSet test) {
        preprocessTrain(test);
    }

}
