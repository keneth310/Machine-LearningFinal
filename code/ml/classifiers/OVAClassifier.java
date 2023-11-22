package ml.classifiers;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;
import ml.data.CrossValidationSet;
import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;

/**
 * Implements the OVA Classifier algorithm.
 * 
 * @authors Brisa Salazar and Kenny Gonzalez
 */

public class OVAClassifier implements Classifier {
	private ClassifierFactory cFactory;

	private HashMap<Double, Classifier> classifierMap = new HashMap<>();; // to store the mini classifier in train

	public OVAClassifier(ClassifierFactory cFactory) {
		this.cFactory = cFactory;
	}

	/**
	 * Train this classifier based on the data set
	 * 
	 * @param data
	 */
	public void train(DataSet data) {
		ArrayList<Example> dataArr = data.getData();
		Set<Double> labelSet = data.getLabels();
		// we do one versus all for each label
		for (Double label : labelSet) {
			DataSet copy = new DataSet(data.getFeatureMap()); // creates a new dataset with the redefined labels
			// we filter the data sets and examples
			for (Example e : dataArr) {
				// check if same label as the one we are classifying on
				Example eCopy = new Example(e);
				if (e.getLabel() == label) {
					eCopy.setLabel(1.0);
				} else {
					eCopy.setLabel(-1.0);
				}
				copy.addData(eCopy);
			}
			Classifier classifier = cFactory.getClassifier();
			classifier.train(copy);
			this.classifierMap.put(label, classifier);// stores the classifier we've worked with
		}
	}

	/**
	 * Classify the example. Should only be called *after* train has been called.
	 * 
	 * @param example
	 * @return the class label predicted by the classifier for this example
	 */
	public double classify(Example example) {
		Double binaryPrediction = 0.0;
		Double minConfidence = null;
		Double maxConfidence = null;
		Double minConfidentLabel = null;
		Double maxConfidentLabel = null;
		boolean posPrediction = false;

		for (Double label : this.classifierMap.keySet()) {
			Double currConfidence = this.classifierMap.get(label).confidence(example);
			binaryPrediction = this.classifierMap.get(label).classify(example);
			if (binaryPrediction > 0){ // if positive want to return the most confident positive
				if (maxConfidence == null || currConfidence > maxConfidence){
					posPrediction = true;
					maxConfidence = currConfidence;
					maxConfidentLabel = label;
				}
			}
			else { // negative so want to return least confident negative 
				if (minConfidence == null || currConfidence < minConfidence){
					minConfidence = currConfidence;
					minConfidentLabel = label;
				}
			}
		}
		// return most confident positive or least confident negative
		if (posPrediction){
			return maxConfidentLabel;
		}
		else {
			return minConfidentLabel;
		}
	}

	/**
	 * Gets confidence of the example
	 * @param example an example from data
	 * @return double with the confidence
	 */
	public double confidence(Example example) {
		return 0.0;
	}

	public static void main(String[] args) {
		DataSet someData = new DataSet("data/wines.train", DataSet.TEXTFILE); // make new dataset with simple vars
		// CrossValidationSet crossValidation = new CrossValidationSet(someData, 10, true);
		ClassifierFactory cl = new ClassifierFactory(0, 3);
		OVAClassifier ovacl = new OVAClassifier(cl);
       	ovacl.train(someData);
		// DecisionTreeClassifier dTree = cl.getClassifier();
		ovacl.toString();
		// Classifier classifier = cl.getClassifier();
		// DecisionTreeClassifier dTree = classifier.;

		// train first 
		// data.get all labels and get 10, 
		// build the tree on that feature, print that one, and then set depth limit of 3
		// 
		

		// run for the number of splits 
	// 	ArrayList<Double> splitAvgs = new ArrayList<Double>();
	// 	for (int i = 0; i < crossValidation.getNumSplits(); i++){
	// 		DataSetSplit dataSplit = crossValidation.getValidationSet(i);
	// 		ovacl.train(dataSplit.getTrain()); // training data
	// 		double allAccuracy = 0.0;
	// 		// runs classifier 100 times to find accuracy 
	// 		for (int j = 0; j < 100; j++) {
	// 			double correctCount = 0.0;
	// 			for (Example e : dataSplit.getTest().getData()) {
	// 				//System.out.println("example: " + e);
	// 				double prediction = ovacl.classify(e);
	// 				if (prediction == e.getLabel()) {
	// 					//System.out.println("in here");
	// 					correctCount += 1;
	// 				}
	// 			}
	// 			double currentAccuracy = correctCount / dataSplit.getTest().getData().size();
	// 			allAccuracy += currentAccuracy;
	// 		}
	// 		double avg = allAccuracy / 100;
	// 		splitAvgs.add(avg); // will hold the avg accuragy from thr ith split 
	// 	}
	// 	System.out.println("splitAvgs: " + splitAvgs);
	// 	// uncode this when you have more than one fold
    //     double sumAvgs = 0.0;
    //     for (int i = 0; i < splitAvgs.size(); i++) {
    //         sumAvgs += splitAvgs.get(i);
    //     }
    //     System.out.println(sumAvgs / crossValidation.getNumSplits());
	// }
	}
}

	