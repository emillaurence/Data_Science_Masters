'''
Created on 7May,2017

@author: User
'''
import datetime
import argparse
import comp5318_assign1_perceptron
import comp5318_assign1_softmax
import comp5318_assign1_naivebayes
import comp5318_assign1_knn

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("-path", "--path", help="File path.", type=str, default="C:\\code\\")
    parser.add_argument("-trndata", "--trainingDataFile", help="Training data file name.", type=str, default='training_data.csv')
    parser.add_argument("-trnlabel", "--trainingLabelData", help="Training label data file name.", type=str, default='training_labels.csv')
    parser.add_argument("-tstlabel", "--testingDataFile", help="Training label data file name.", type=str, default='test_data.csv')
    parser.add_argument("-kfold", "--kfold", help="Number of Folds in Cross Validation.", type=int, default=10)
    parser.add_argument("-cv", "--crossvalidation", help="Cross Validation Enabled or Disabled.  There is no cross validatin for knn and naive bayes", type=bool, default=False)
    parser.add_argument("-testFile", "--testingLabelFile", help="Output file of the prediction label", type=str, default="predicted_labels.csv")
    parser.add_argument("-c", "--classifier", help="Can have the any of the values: softmax, perceptron, naive, knn. Default is softmax", type=str, default="softmax")

    # Parse arguments
    args = parser.parse_args()

    return args



if __name__ == '__main__':
    start = datetime.datetime.now()
    # Parse the arguments
    args = parseArguments()

    # Raw print arguments
    print("You are running the script with arguments: ")
    for a in args.__dict__:
        print(str(a) + ": " + str(args.__dict__[a]))
    
    # Initialise Variables
    path_data = args.path
    trainingDataFile = args.trainingDataFile 
    trainingLabelData = args.trainingLabelData 
    testingDataFile = args.testingDataFile 
    testingLabelFile = args.testingLabelFile
    classifier = args.classifier
    nFolds = args.kfold

    if (classifier == "perceptron"):
        comp5318_assign1_perceptron.runPerceptron(path_data, trainingDataFile, trainingLabelData, testingDataFile, testingLabelFile, args.crossvalidation, nFolds)
    elif (classifier == "softmax"):
        comp5318_assign1_softmax.runSoftmaxLogistic(path_data, trainingDataFile, trainingLabelData, testingDataFile, testingLabelFile, args.crossvalidation, nFolds)
    elif (classifier == "naive"): 
        comp5318_assign1_naivebayes.runNaiveBayes(path_data, trainingDataFile, trainingLabelData, testingDataFile, args.crossvalidation, nFolds)
    elif (classifier == "knn"):
        comp5318_assign1_knn.runKNN(path_data, trainingDataFile, trainingLabelData, testingDataFile, args.crossvalidation, nFolds)
    else:  
        print ("Invalid classifier given:", classifier)

    end = datetime.datetime.now()
    print("Completed in", (end-start))