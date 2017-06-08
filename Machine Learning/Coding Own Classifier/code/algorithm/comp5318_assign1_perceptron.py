import datetime
import os
import numpy as np
from DataLibrary import PerceptronDataReader, TFIDFInputParser
from Model import PerceptronClassifier
from ModelValidation import MyKFold
from Metrics import Scores


#runs the perceptron classifier.  Includes reading of data, cleaning, data munging/parsing,
#cross validation, classifier evaluation, score/metrics 
def runPerceptron(path_data, trainingDataFile, trainingLabelData, testingDataFile, testingLabelFile, crossvalidation, nFolds, isSaveTestLabel=False):
    a = datetime.datetime.now()
    
    isTest = crossvalidation == False
    
    trainingDataFile = os.path.join(path_data, os.path.join("input", trainingDataFile) )
    trainingLabelData = os.path.join(path_data, os.path.join("input", trainingLabelData) )
    testingDataFile = os.path.join(path_data, os.path.join("input", testingDataFile) )
    testingLabelFile = os.path.join(path_data, os.path.join("output", testingLabelFile) )
    
    print("running perceptron classification, using 1000 features") 
    reader = PerceptronDataReader(1000, trainingDataFile, trainingLabelData, testingDataFile)
    reader.initTrainingData()
    reader.readTestFile()
    
    print("\n Preparing data for classifier ")
    minput = TFIDFInputParser(reader)
    classifier = PerceptronClassifier(10, minput, )
    
    if (isTest):
        train_index = [[i for i in range(reader.totalApps)]]
        test_index = [[i for i in range(len(reader.testData))]]
        nFolds = 1
    else:
        train_index, test_index = MyKFold().getFolds(minput.getData(), nFolds, True) 

    y_predict, y_test = [], []
    overallaccuracy = 0
    
    for i in range(1, nFolds+1):
        train = train_index[i-1]  
        test = test_index[i-1]
        print("evaluate classifier on %d fold.  Total train data: %d.  Total test data: %d" % ( i, len(train), len(test) ))
        predicted = classifier.classify(train, test)
        
        if (isTest):
            if (isSaveTestLabel):                     
                predictions = []
                for index in test: predictions.append([minput.testTitles[index], predicted[index]])
                print("Saving predictions to", testingLabelFile)
                pa = np.asarray(predictions)
                np.savetxt(testingLabelFile, pa, delimiter=",", fmt="%s")
        else:
            
            correct = 0
            #check accuracy, and save test
            for j, p in enumerate(predicted):
                index = test[j]
                actual_label = minput.getLabel(index)
                y_test.append(actual_label)
                y_predict.append(p)
                if actual_label == p: correct += 1
    
            accuracy = correct / float(len(test)) * 100.0
            print("%s fold has an accuracy score of %s" % (i, accuracy))
            overallaccuracy += accuracy
        
    if not isTest: 
        print("overall accuracy is", overallaccuracy/nFolds)
        Scores().getScores(minput.getLabels(), y_test, y_predict)


    b = datetime.datetime.now()
    print("\n*****  ENDED *****", b-a)
    
