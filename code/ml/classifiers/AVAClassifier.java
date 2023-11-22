package ml.classifiers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import ml.data.CrossValidationSet;
import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;

/**
 * Implements the AVA Classifier algorithm.
 * 
 * @author Brisa Salazar and Kenny Gonzalez
 */

public class AVAClassifier implements Classifier {
	private ClassifierFactory cFactory;
	private HashMap<String, Classifier> clPairs = new HashMap<>();

	public AVAClassifier(ClassifierFactory cFactory) {
		this.cFactory = cFactory;
	}

	/**
	 * Train this classifier based on the data set
	 * 
	 * @param data
	 */
	public void train(DataSet data) {
		ArrayList<Example> dataArr = data.getData();
		int numLabels = data.getLabels().size();
		
		for (int i = 0; i < numLabels; i++){
			for (int k = i + 1; k < numLabels; k++){
				Classifier classifier = cFactory.getClassifier();
				DataSet copy = new DataSet(data.getFeatureMap());
				for (Example e : dataArr){
					// check if same label as the one we are classifying on
					Example eCopy = new Example(e);
					if (e.getLabel() == i) {
						eCopy.setLabel(1.0);
						copy.addData(eCopy);
					} else if (e.getLabel() == k) {
						eCopy.setLabel(-1.0);
						copy.addData(eCopy);
					}
				}
				// storing the classifier and label pairs 
				classifier.train(copy);
				clPairs.put(i + "v" + k, classifier); // string is the label, classifier is the classifier
			}
		}
	}

	/**
	 * Classify the example. Should only be called *after* train has been called.
	 * 
	 * @param example
	 * @return the class label predicted by the classifier for this example
	 */
	public double classify(Example example) {
		HashMap<Double, Double> runningLabelHash =  new HashMap<>();
		for (String label : this.clPairs.keySet()){
			// split the label pair to get the individiual labels 
			String[] labels = label.split("v");
			Double i = Double.parseDouble(labels[0]); 
			Double k = Double.parseDouble(labels[1]);
	
			Double prediction = clPairs.get(label).classify(example); // gets the classifier tied to that label, classifies using classifier 
			Double confidence = clPairs.get(label).confidence(example); // gets the classifier tied to that label, gives us the confidence using classifier
			Double weightedConfidence = prediction * confidence;
			
			// if the label i, k is in then returns that value in the hash. else if no value is there, sets to 0.0 as a default. 
			Double iScore = runningLabelHash.getOrDefault(i, 0.0);
			Double kScore = runningLabelHash.getOrDefault(k, 0.0);
			
			runningLabelHash.put(i,iScore + weightedConfidence);
			runningLabelHash.put(k,kScore - weightedConfidence);
			}
		// find the label with the greatest confidence
		return Collections.max(runningLabelHash.entrySet(), Map.Entry.comparingByValue()).getKey();
	}


	public double confidence(Example example) {
		return 0.0;
	}
	public static void main(String[] args) {
		DataSet someData = new DataSet("data/wines.train", DataSet.TEXTFILE); // make new dataset with simple vars
		CrossValidationSet crossValidation = new CrossValidationSet(someData, 10, true);
		ClassifierFactory cl = new ClassifierFactory(0,3);
		AVAClassifier avacl = new AVAClassifier(cl);

       
		// run for the number of splits 
		ArrayList<Double> splitAvgs = new ArrayList<Double>();
		for (int i = 0; i < crossValidation.getNumSplits(); i++){
			DataSetSplit dataSplit = crossValidation.getValidationSet(i);
			avacl.train(dataSplit.getTrain()); 
			double allAccuracy = 0.0;
			// runs classifier 100 times to find accuracy 
			for (int j = 0; j < 100; j++) {
				double correctCount = 0.0;
				for (Example e : dataSplit.getTest().getData()) {
					double prediction = avacl.classify(e);
					if (prediction == e.getLabel()) {
						correctCount += 1;
					} 
				}
				double currentAccuracy = correctCount / dataSplit.getTest().getData().size();
				allAccuracy += currentAccuracy;
			}
			double avg = allAccuracy / 100;
			splitAvgs.add(avg); // will hold the avg accuragy from thr ith split 
		}
	
        double sumAvgs = 0.0;
        for (int i = 0; i < splitAvgs.size(); i++) {
            sumAvgs += splitAvgs.get(i);
        }
        System.out.println(sumAvgs / crossValidation.getNumSplits());
	}
}
