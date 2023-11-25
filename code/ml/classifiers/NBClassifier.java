package ml.classifiers;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Comparator;
import java.util.Collections;
import ml.data.CrossValidationSet;
import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;

/**
* A classifier that implements the Naive Bayes machine learning algorithm. 
*
* @author Brisa Salazar and Keneth Gonzalez 
*
*/

public class NBClassifier implements Classifier {
    private double lambda;
    private HashMap<Double, HashMap<Integer, Integer>> labelFeaturesCount = new HashMap<>();  
    private HashMap<Double, Integer> labelCount = new HashMap<>(); 
    private boolean usePosFeatures = false;
    private DataSet data; 

    /**
     * Constructor for the class
     */
    public NBClassifier(){}

    /**
     * Sets the lamda for the classifier
     * @param x
     */
    public void setLambda(double x){
        this.lambda = x;
    }

    /**
     * Chooses between the different classification variants.
     * If true, should only use positive features. 
     * By default, it is set to false (uses all features).
     * @param bool
     */
    public void setUseOnlyPositiveFeatures(boolean bool){
        this.usePosFeatures = bool;
    }

    /**
	 * Train this classifier based on the data set
     * Should retrain the model from scratch each time
	 * 
	 * @param data
	 */
	public void train(DataSet data){
        ArrayList<Example> examples = data.getData(); 
        this.data = data; 
        Collections.shuffle(examples);
        
        // for all labels, create a 2 hashmaps, the count and feature count
        for (Double label : data.getLabels()){ 

            // we want to add the label to label count and set the value to 0 for all labels
            this.labelCount.put(label, 0); 

            // we want to add all the labels to the label features count
            this.labelFeaturesCount.put(label, new HashMap<>());
        }
        

        // go through all examples in data
        for (Example example: examples){

            // udpate label's counts
            double currLabel = example.getLabel();
            int currCount = this.labelCount.get(currLabel); 
            currCount += 1; 
            this.labelCount.put(currLabel, currCount);
            
            // save features that appear with that label in that example
            for (Integer feature : example.getFeatureSet()){
                HashMap<Integer, Integer> featureCount = this.labelFeaturesCount.get(currLabel);

                // if feature is not in the hashmap 
                if (!featureCount.containsKey(feature)){
                    featureCount.put(feature, 1);
                }

                // if feature is in the hashmap 
                else{
                    int count = featureCount.get(feature); 
                    count += 1; 
                    featureCount.put(feature, count); 
                }

                // update the hashmap of that label 
                this.labelFeaturesCount.put(currLabel, featureCount);
            }
        }
        System.out.println("here is label count: " + labelCount);
        System.out.println("here is label feature count: " + labelFeaturesCount);
    }

    /**
     * Return the log base probabilty of the example with the label under 
     * the current trained model
     * @param ex
     * @param label
     * @return 
     */
    public double getLogProb(Example ex, double label){
       double labelProb;
       double logProb;
       double sum = 0.0; 

       // how many times that label appears in the examples
       double numerator = this.labelCount.get(label);

       // how many examples we're working with
       float denominator = this.data.getData().size();
       labelProb = numerator / denominator;

       if (this.usePosFeatures){ 
        
            // we want to loop through all of feature in the example (present features), and get logprob of that
            for (int featureIndex : ex.getFeatureSet()){
                if (ex.getFeature(featureIndex) > 0) {
                    sum += Math.log10(getFeatureProb(featureIndex, label));
                }
            }
            logProb = Math.log10(labelProb) + sum;
        
       } else {
        
            // we want to loop through all possible features for that label(including those not in the example) and determine the probability 
            for (int featureIndex : data.getAllFeatureIndices()){
                
                // if the feature is in the example
                if (ex.getFeatureSet().contains(featureIndex) && ex.getFeature(featureIndex) > 0){
                    sum += Math.log10(getFeatureProb(featureIndex, label));
                }

                // if feature not in example 
                else{
                    sum += Math.log10(1 - getFeatureProb(featureIndex, label));
                }
            }
            logProb = Math.log10(labelProb) + sum;
       }
       return logProb;
    }

    /**
     * Should give p(xi|y) for the model
     * @param featureIndex
     * @param label
     * @return
     */
    public double getFeatureProb(int featureIndex, double label){
        HashMap<Integer, Integer> features = labelFeaturesCount.get(label);  
        double numerator = 0.0; 

        // check if feature is in the feature count 
        if (features.containsKey(featureIndex)) { 
            numerator = features.get(featureIndex); 
        }
        
        // count how many times that label appears
        double denominator = this.labelCount.get(label); 
      
        // smoothed probability
        return (numerator + this.lambda) /(denominator + (2*this.lambda));
    }

	/**
	 * Classify the example.  Should only be called *after* train has been called.
	 * 
	 * @param example
	 * @return the class label predicted by the classifier for this example
	 */
	public double classify(Example example){
        double maxProb = -Double.MAX_VALUE; 
        double prediction = 0.0; 

        // for all of the possible labels, we find probability for that example
        for (Double label : data.getLabels()){ 
            double logProb = getLogProb(example, label); 
            // System.out.println("current label" + label + "current log prob: " + logProb);
            // keep track of max probability and label for that
            if (logProb > maxProb){ 
                maxProb = logProb; 
                prediction = label; 
            }
        }
        
        return prediction; 
    }
	
    /**
     * Return the log probability of the most likely label
     * @return the log probability 
     */
	public double confidence(Example example){
        double label = classify(example);
        return getLogProb(example, label);
    }

    /**
     * Main method for testing 
     */
    public static void main(String[] args){
        NBClassifier cl = new NBClassifier(); 
		DataSet data = new DataSet("data/diabetesDecimalLabel.csv", 0); // 0 for csv file 
        // CrossValidationSet cv = new CrossValidationSet(data, 10);
        // DataSetSplit dataSplit = cv.getValidationSet(1);
        // System.out.println("dataSetSplit size:" + dataSplit.getTest().getData().size());
        DataSetSplit dataSplit = data.split(.8);
        cl.setUseOnlyPositiveFeatures(true);
        cl.setLambda(0.01);
        cl.train(dataSplit.getTrain());
        System.out.println("the labels: " + dataSplit.getTest().getLabels());

        // from test set 
        // Example first = dataSplit.getTest().getData().get(0);

        // System.out.println("the example: " + first);
        // System.out.println("the label: " + first.getLabel());
        // System.out.println("the prediction: " + someClassifier.classify(first));

        ArrayList<Example> arrData = dataSplit.getTrain().getData();

        double correctCount = 0.0;
        for (Example e : arrData) {
            //System.out.println("example: " + e);
            double prediction = cl.classify(e);
            if (prediction == e.getLabel()) {
                //System.out.println("in here");
                correctCount += 1;
            } 
            // else { 
            //     System.out.print("wrong prediction: " + prediction + "\n"); 
            // }
        }
        double currentAccuracy = correctCount / arrData.size();
        System.out.println("the accuracy: " + currentAccuracy);



        // *** EXPERIMENTS- determining best lambda accuracy ***
        // ArrayList<ArrayList<Double>> arr = new ArrayList<>();
        
        // for (Example e : dataSplit.getTest().getData()) {
        //     double prediction = someClassifier.classify(e);
        //     double Confidence = someClassifier.confidence(e); 

        //     // add confidence and prediction to array
        //     ArrayList<Double> innerArr = new ArrayList<>(); 
        //     // innerArr will have [confidence, predictii]
        //     innerArr.add(Confidence);
        //     innerArr.add(prediction);
        //     innerArr.add(e.getLabel());
        //     arr.add(innerArr);
        // }
       
        // // sort the pairs by confidence 
        // Collections.sort(arr, new Comparator<ArrayList<Double>>() {
        //     @Override
        //     public int compare(ArrayList<Double> list1, ArrayList<Double> list2) {
        //         // Comparing based on the first element of each inner list
        //         return list1.get(0).compareTo(list2.get(0));
        //     }
        // });

        // // reverse so that it is in decreasing order
        // Collections.reverse(arr); 
        // //System.out.println(arr); 
        

        //     double runningAcc = 0.0;
        //     double countSeen = 0.0;
        //     double correctCount = 0.0;
        //     ArrayList<ArrayList<Double>> confidenceAccuracies = new ArrayList<>();
        //     // loop through each confidence from greatest to smallest
        //     for (ArrayList<Double> subArr : arr){
        //         ArrayList<Double> mini = new ArrayList<>();
        //         countSeen += 1.0;
        //         double pred = subArr.get(1); 
        //         double actual = subArr.get(2); 
        //         if (pred == actual){
        //             correctCount += 1.0;
        //         }
        //         // System.out.println("correctCount: " + correctCount);
        //         // System.out.println("count seen: " + countSeen);
        //         runningAcc = correctCount / countSeen;
        //         mini.add((double) subArr.get(0)); // confidence
        //         mini.add(runningAcc); // running accuracy 
        //         confidenceAccuracies.add(mini);
        //        // System.out.print(mini + "\n");

        //     }
         }       
        
        
    }



    

