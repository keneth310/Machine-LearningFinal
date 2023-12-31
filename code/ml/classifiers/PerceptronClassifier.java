package ml.classifiers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Set;
import java.util.Random;

import ml.data.CrossValidationSet;
import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;
import ml.data.ExampleNormalizer;
import ml.data.FeatureNormalizer;

/**
 * Basic perceptron classifier
 * 
 * @author dkauchak
 *
 */
public class PerceptronClassifier implements Classifier {
	protected HashMap<Integer, Double> weights; // the feature weights
	protected double b = 0; // the intersect weight
	
	protected int iterations = 10;
		
	/**
	 * Get a weight vector over the set of features with each weight
	 * set to 0
	 * 
	 * @param features the set of features to learn over
	 * @return
	 */
	protected HashMap<Integer, Double> getZeroWeights(Set<Integer> features){
		HashMap<Integer, Double> temp = new HashMap<Integer, Double>();
		
		for( Integer f: features){
			temp.put(f, 0.0);
		}
		
		return temp;
	}
	
	/**
	 * Initialize the weights and the intersect value
	 * 
	 * @param features
	 */
	protected void initializeWeights(Set<Integer> features){
		weights = getZeroWeights(features);
		b = 0;
	}
	
	/**
	 * Set the number of iterations the perceptron should run during training
	 * 
	 * @param iterations
	 */
	public void setIterations(int iterations){
		this.iterations = iterations;
	}
	
	public void train(DataSet data) {
		initializeWeights(data.getAllFeatureIndices());
		
		ArrayList<Example> training = (ArrayList<Example>)data.getData().clone();
		
		for( int it = 0; it < iterations; it++ ){
			Collections.shuffle(training);
			
			for( Example e: training ){
				if( getPrediction(e) != e.getLabel() ){
					double label = e.getLabel();
					
					// update the weights
					//for( Integer featureIndex: weights.keySet() ){
					for( Integer featureIndex: e.getFeatureSet() ){
						double oldWeight = weights.get(featureIndex);
						double featureValue = e.getFeature(featureIndex);
						
						weights.put(featureIndex, oldWeight + featureValue*label);
					}
					
					// update b
					b += label;					
				}
			}
		}
		// System.out.println("the weights are: " + weights);
		// System.out.println("the bias is: " + b);
	}

	@Override
	public double classify(Example example) {
		return getPrediction(example);
	}
	
	@Override
	public double confidence(Example example) {
		return Math.abs(getDistanceFromHyperplane(example, weights, b));
	}

		
	/**
	 * Get the prediction from the current set of weights on this example
	 * 
	 * @param e the example to predict
	 * @return
	 */
	protected double getPrediction(Example e){
		return getPrediction(e, weights, b);
	}
	
	/**
	 * Get the prediction from the on this example from using weights w and inputB
	 * 
	 * @param e example to predict
	 * @param w the set of weights to use
	 * @param inputB the b value to use
	 * @return the prediction
	 */
	protected static double getPrediction(Example e, HashMap<Integer, Double> w, double inputB){
		double sum = getDistanceFromHyperplane(e,w,inputB);

		if( sum > 0 ){
			return 1.0;
		}else if( sum < 0 ){
			return -1.0;
		}else{
			return 0;
		}
	}
	
	protected static double getDistanceFromHyperplane(Example e, HashMap<Integer, Double> w, double inputB){
		double sum = inputB;
		
		//for(Integer featureIndex: w.keySet()){
		// only need to iterate over non-zero features
		for( Integer featureIndex: e.getFeatureSet()){
			sum += w.get(featureIndex) * e.getFeature(featureIndex);
		}
		
		return sum;
	}
	
	public String toString(){
		StringBuffer buffer = new StringBuffer();
		
		ArrayList<Integer> temp = new ArrayList<Integer>(weights.keySet());
		Collections.sort(temp);
		
		for(Integer index: temp){
			buffer.append(index + ":" + weights.get(index) + " ");
		}
		
		return buffer.substring(0, buffer.length()-1);
	}

	public static void main(String[] args){ 
		PerceptronClassifier cl = new PerceptronClassifier(); 
		DataSet someData = new DataSet("data/diabetesDecimalLabel.csv", 0); 
		CrossValidationSet crossValidation = new CrossValidationSet(someData, 10, true);
	
		// run for the number of splits 
		ArrayList<Double> splitAvgs = new ArrayList<Double>();
		for (int i = 0; i < crossValidation.getNumSplits(); i++){
			DataSetSplit dataSplit = crossValidation.getValidationSet(i);
			//preproces data
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
			break;
		}
		System.out.println(splitAvgs);
        // double sumAvgs = 0.0;
        // for (int i = 0; i < splitAvgs.size(); i++) {
        //     sumAvgs += splitAvgs.get(i);
        // }
        // System.out.println(sumAvgs / crossValidation.getNumSplits());	
		// }
	}
}

