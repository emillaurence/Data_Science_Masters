'''
Created on 7May,2017

@author: User
'''
import os
import numpy as np

from DataLibrary import DataReader, PreProcessing, FeatureSelection
from Model import BayesNaiveClassifier
from ModelValidation import CrossValidation
from Metrics import classificationReport

#runs the Naive Bayes classifier.  Includes reading of data, cleaning, data munging/parsing,
#cross validation, classifier evaluation, score/metrics 
def runNaiveBayes(path_data, trainingDataFile, trainingLabelData, testingDataFile, crossvalidation, kfold):
    path_data = os.path.join(path_data, "input") 
    
    if (crossvalidation): print("There is no cross validation for Naive Bayes.")
    # Read Training Data
    training = DataReader()
    
    classLabelEncoder, trainingData = training.processData(path_data, trainingDataFile, dataLabelFile = trainingLabelData)
    
    # Pre-processing Training Data
    preProcess = PreProcessing()
    
    # Removing Rows with Zero values
    trainingData = preProcess.removeZeroRow(trainingData, 0, len(trainingData[0])-1)
    
    n_components = 400
    
    fs =   FeatureSelection()
     
    trainingData = fs.FeatureSelection(trainingData, 1)
    
    # Shuffle Training Data
    np.random.shuffle(trainingData)

    # Split the data into the X and Y component
    X_input, Y_target = preProcess.extractXY(trainingData)

    # Split the training data into train and test
    train, test = preProcess.trainTestSplit(trainingData, 10)
    X_train, Y_train = preProcess.extractXY(train)
    X_test, Y_test = preProcess.extractXY(test)
    
    
    classifier = BayesNaiveClassifier(ColCount=X_train.shape[1])
    classifier.fit(X_train, Y_train)

    Y_predict = classifier.predict(X_test)
    
    
    # Accuracy
    report = classificationReport()
    print('Classifier Accuracy:', report.accuracyScore(Y_test, Y_predict))
    
