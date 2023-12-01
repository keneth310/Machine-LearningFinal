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
 * Gradient descent classifier allowing for two different loss functions and
 * three different regularization settings.
 * 
 * @author dkauchak and Brisa Salazar and Kenneth Gonzalez
 *
 */
public class GradientDescentClassifier implements Classifier {
	// constants for the different surrogate loss functions
	public static final int EXPONENTIAL_LOSS = 0;
	public static final int HINGE_LOSS = 1;

	// constants for the different regularization parameters
	public static final int NO_REGULARIZATION = 0;
	public static final int L1_REGULARIZATION = 1;
	public static final int L2_REGULARIZATION = 2;

	protected HashMap<Integer, Double> weights; // the feature weights
	protected double b = 0; // the intersect weight

	protected int iterations = 10;

	private double lamda = 0.01;;
	private double eta = 0.01;
	private int chosenLoss = 0;
	private int chosenRegularization = 0;

	/**
	 * Constructor of GradientDescentClassifier
	 */
	public GradientDescentClassifier() {
	}

	/**
	 * Takes an int and selects the loss function to use
	 * 
	 * @param int
	 */
	public void setLoss(int x) {
		// soemthing here what does (based on constants mean here)
		if (x == EXPONENTIAL_LOSS) {
			this.chosenLoss = 0; // will be an exponential loss function
		} else {
			this.chosenLoss = 1; // hinge los function
		}
	}

	/**
	 * Takes an int and selects the regularization method to use
	 * 
	 * @param x
	 */
	public void setRegularization(int x) {
		// something here, same as comment for setLoss
		if (x == L2_REGULARIZATION) {
			this.chosenRegularization = 2; // L2 regularization
		} else if (x == L1_REGULARIZATION) {
			this.chosenRegularization = 1; // L1 regularization
		} else {
			this.chosenRegularization = 0; // NO regularization
		}
	}

	/**
	 * Takes a double and sets the lamda to use
	 * 
	 * @param double
	 */
	public void setLamda(double x) {
		this.lamda = x;
	}

	/**
	 * Takes a double and sets the eta to use
	 * 
	 * @param double
	 */
	public void setEta(double x) {
		this.eta = x;
	}

	/**
	 * Get a weight vector over the set of features with each weight
	 * set to 0
	 * 
	 * @param features the set of features to learn over
	 * @return
	 */
	protected HashMap<Integer, Double> getZeroWeights(Set<Integer> features) {
		HashMap<Integer, Double> temp = new HashMap<Integer, Double>();

		for (Integer f : features) {
			temp.put(f, 0.0);
		}

		return temp;
	}

	/**
	 * Initialize the weights and the intersect value
	 * 
	 * @param features
	 */
	protected void initializeWeights(Set<Integer> features) {
		weights = getZeroWeights(features);
		b = 0;
	}

	/**
	 * Performs the regularization depending on what the regularizer is
	 * 
	 * @param chosenRegulization
	 * @return
	 */
	private double regularize(int chosenRegulization, double currentWeight) {
		if (chosenRegularization == L1_REGULARIZATION) {
			// implement L1 regularization
			return Math.signum(currentWeight);
		} else if (chosenRegularization == L2_REGULARIZATION) {
			// implement L2 regularization
			return currentWeight;
		} else { // no regularization
			return 0.0;
		}
	}

	/**
	 * Performs the loss function depending on what the function to be used is
	 * 
	 * @param chosenLoss
	 * @return
	 */
	private double lossFunc(int chosenLoss, double label, double prediction) {
		double c = 0.0;
		if (chosenLoss == EXPONENTIAL_LOSS) { // implement exponential fucntion
			c = Math.exp((-1 * label) * (prediction));
		} else { // implement hinge loss function here
			if (label * prediction < 1) {
				c = 1;
			} else {
				c = 0;
			}
		}
		return c;
	}

	/**
	 * Set the number of iterations the perceptron should run during training
	 * 
	 * @param iterations
	 */
	public void setIterations(int iterations) {
		this.iterations = iterations;
	}

	public void train(DataSet data, int featureToRemove) {
		initializeWeights(data.getAllFeatureIndices());

		ArrayList<Example> training = (ArrayList<Example>) data.getData().clone();
		for (int it = 0; it < iterations; it++) {
			Collections.shuffle(training);
			double lossSum = 0.0;

			for (Example e : training) {
				e.removeFeature(featureToRemove);
				double label = e.getLabel();
				double prediction = getDistanceFromHyperplane(e, weights, b);
				lossSum += calcLoss(this.chosenLoss, e.getLabel(), prediction);

				for (Integer featureIndex : e.getFeatureSet()) {
						double oldWeight = weights.get(featureIndex);
						double featureValue = e.getFeature(featureIndex);
						double newWeight = oldWeight
								+ this.eta * ((label * featureValue * lossFunc(this.chosenLoss, label, prediction))
										- (lamda * regularize(this.chosenRegularization, oldWeight)));
						weights.put(featureIndex, newWeight);
				}

				b += this.eta * ((label * 1 * lossFunc(this.chosenLoss, label, prediction))
						- (this.lamda * regularize(this.chosenRegularization, b)));
			}
		}
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
	protected double getPrediction(Example e) {
		return getPrediction(e, weights, b);
	}

	/**
	 * Get the prediction from the on this example from using weights w and inputB
	 * 
	 * @param e      example to predict
	 * @param w      the set of weights to use
	 * @param inputB the b value to use
	 * @return the prediction
	 */
	protected static double getPrediction(Example e, HashMap<Integer, Double> w, double inputB) {
		double sum = getDistanceFromHyperplane(e, w, inputB);

		if (sum > 0) {
			return 1.0;
		} else if (sum < 0) {
			return -1.0;
		} else {
			return 0;
		}
	}

	protected static double getDistanceFromHyperplane(Example e, HashMap<Integer, Double> w, double inputB) {
		double sum = inputB;

		// for(Integer featureIndex: w.keySet()){
		// only need to iterate over non-zero features
		for (Integer featureIndex : e.getFeatureSet()) {
			sum += w.get(featureIndex) * e.getFeature(featureIndex);
		}

		return sum;
	}

	/***
	 * Calcualtes loss (non-derivative)
	 * 
	 * @param chosenLoss
	 * @param label
	 * @param prediction
	 * @return
	 */
	private double calcLoss(int chosenLoss, double label, double prediction) {
		double c = 0.0;
		if (chosenLoss == EXPONENTIAL_LOSS) { // implement exponential fucntion
			c = Math.exp((-1 * label) * (prediction));
		} else { // implement hinge loss function here
			c = Math.max(0, 1 - (label * prediction));
		}
		return c;

	}

	public String toString() {
		StringBuffer buffer = new StringBuffer();

		ArrayList<Integer> temp = new ArrayList<Integer>(weights.keySet());
		Collections.sort(temp);

		for (Integer index : temp) {
			buffer.append(index + ":" + weights.get(index) + " ");
		}

		return buffer.substring(0, buffer.length() - 1);
	}

	/**
	 * Main function to test gradient descent algorithm
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		DataSet data = new DataSet("data/diabetesDecimalLabel.csv", 0);
		// Collins is cooking
		int totalFeatures = data.getAllFeatureIndices().size();
		List<Double> accuracies = new ArrayList<Double>();

		for (Integer indexToRemove: data.getFeatureMap().keySet()){ 
			DataSet someCopy = new DataSet(data.getFeatureMap()); // copy of the original data set 
			GradientDescentClassifier gdescent = new GradientDescentClassifier();
			gdescent.train(someCopy, indexToRemove);
			
		}
		
		




		







		CrossValidationSet crossValidation = new CrossValidationSet(data, 10, true);
		gdescent.setLoss(HINGE_LOSS);
		gdescent.setIterations(30);
		// gdescent.setEta(0.01);
		// gdescent.setLamda(0.05);
		gdescent.setRegularization(NO_REGULARIZATION);

		// run for the number of splits
		ArrayList<Double> splitAvgs = new ArrayList<Double>();
		for (int i = 0; i < crossValidation.getNumSplits(); i++) {
			DataSetSplit dataSplit = crossValidation.getValidationSet(i);

			// preprocess data
			FeatureNormalizer featureNormalizer = new FeatureNormalizer();
			ExampleNormalizer exampleNormalizer = new ExampleNormalizer();
			featureNormalizer.preprocessTrain(dataSplit.getTrain());
			exampleNormalizer.preprocessTrain(dataSplit.getTrain());

			gdescent.train(dataSplit.getTrain());
			double allAccuracy = 0.0;

			// runs classifier 100 times to find accuracy
			for (int j = 0; j < 100; j++) {
				double correctCount = 0.0;
				for (Example e : dataSplit.getTrain().getData()) {
					double prediction = gdescent.classify(e);
					if (prediction == e.getLabel()) {
						correctCount += 1;
					}
				}
				double currentAccuracy = correctCount / dataSplit.getTrain().getData().size();
				allAccuracy += currentAccuracy;
			}
			double avg = allAccuracy / 100;
			splitAvgs.add(avg); // will hold the avg accuragy from thr ith split
			break; // remove when want to get total fold averages
		}
		System.out.println(splitAvgs);
		// double sumAvgs = 0.0;
		// for (int i = 0; i < splitAvgs.size(); i++) {
		// sumAvgs += splitAvgs.get(i);
		// }
		// System.out.println(sumAvgs / crossValidation.getNumSplits());
		// }
	}
}
