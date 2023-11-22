package ml.classifiers;

import java.util.ArrayList;

import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;
 /***
  * @author Brisa Salazar & Keneth Gonzalez 
  */

public class BlastFromPast {
    /**
     * Executes part 4, question 1 of the Assignment Handout 
     * @param args
     */
    public static void main(String[] args){
        DataSet wineDataSet = new DataSet("data/wines.train", DataSet.TEXTFILE);
        DecisionTreeClassifier dTree = new DecisionTreeClassifier();
        //set depth limit to 5
        dTree.setDepthLimit(5);
        // for question 1
        // dTree.train(wineDataSet);
        // System.out.println(dTree);

        // for testing and training accuracy
        double allAccuracyTrain = 0.0;
        double allAccuracyTest = 0.0;
        DataSetSplit dataSplit = wineDataSet.split(.8);
        dTree.train(dataSplit.getTest());
        dTree.train(wineDataSet);
        for (int j = 0; j < 100; j++) {
            ArrayList<Example> examplesTrain = dataSplit.getTrain().getData();
            ArrayList<Example> examplesTest = dataSplit.getTest().getData();
            //ArrayList<Example> examplesTest = wineDataSet.getData();
            double correctCountTrain = 0.0;
            double correctCountTest = 0.0;
			for (Example e : examplesTrain) {
				//System.out.println("example: " + e);
				double prediction = dTree.classify(e);
				if (prediction == e.getLabel()) {
					//System.out.println("in here");
                    correctCountTrain += 1;
                }
			}
            for (Example e : examplesTest) {
				//System.out.println("example: " + e);
				double prediction = dTree.classify(e);
				if (prediction == e.getLabel()) {
					//System.out.println("in here");
                    correctCountTest += 1;
                }
			}

            double currentAccuracyTrain = correctCountTrain / examplesTrain.size();
            double currentAccuracyTest = correctCountTest / examplesTest.size();
            allAccuracyTrain += currentAccuracyTrain;
            allAccuracyTest += currentAccuracyTest;
            

        }
        double avgTrain= allAccuracyTrain / 100;
        double avgTest= allAccuracyTest / 100;
        System.out.println("avgTrain accuracy:" + avgTrain);
        System.out.println("avgTest accuracy:" + avgTest);
    }
}
