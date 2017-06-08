# Authors: Beulah Shantini
#          Emil Laurence Pastor
#          Maureen Rocas
# Course : COMP5318 - Machine Learning and Data Mining

# Import Python Libraries
import numpy as np
import os
from DataLibrary import DataReader, PreProcessing
from Model import SoftmaxLogisticRegression
from ModelValidation import CrossValidation
from Metrics import classificationReport


#runs the Softmax Logistic Regression classifier.  Includes reading of data, cleaning, data munging/parsing,
#cross validation, classifier evaluation, score/metrics 
def runSoftmaxLogistic(path_data, trainingDataFile, trainingLabelData, testingDataFile, testingLabelFile, crossvalidation, kfold):
    
    input_path_data = os.path.join(path_data, "input") 
    
    # Read Training Data
    training = DataReader()
    
    # This training.processData method does the following:
    # 1. Read the training data file
    # 2. Read the training label data file
    # 3. Join the training data and training label data
    # 4. Call the label encoder to translate the categorical variable
    #    into its numerical equivalent.
    classLabelEncoder, trainingData = training.processData(input_path_data, trainingDataFile, dataLabelFile = trainingLabelData)

    # Read Test Data
    testing = DataReader()
    
    # This testing.processData method does the following:
    # 1. Read the testing data file
    testingData = testing.processData(input_path_data, testingDataFile)
    
    # Pre-processing Training Data
    preProcess = PreProcessing()
    
    # Removes the row with Zero values that is considered as a noise
    trainingData = preProcess.removeZeroRow(trainingData, 0, len(trainingData[0])-1)
    
    # Add a dummy column with default value of -1 to align with the dimension of the training data
    testLabel = np.ones((testingData.shape[0],1)) * -1
    testingData = np.column_stack((testingData,testLabel))
    
    # Combine Training and Test Data
    allData = np.vstack((trainingData, testingData))
    
    # Perform Latent Semantic Analysis to reduce the dimension into 500 components
    lsa = preProcess.LSA(500)
    allData = preProcess.LSAFitTransform(lsa,allData,True)
    
    # Separate the training and test data
    trainingData = allData[np.where(allData[:,-1:] != -1)[0],:]
    testingData = allData[np.where(allData[:,-1:] == -1)[0],:-1]

    # Shuffle Training Data to ensure randomness
    np.random.shuffle(trainingData)

    # Split the data into the X and Y component
    X_input, Y_target = preProcess.extractXY(trainingData)

    # Split the training data further into 90% train and 10% test
    print('Splitting to train and test')
    train, test = preProcess.trainTestSplit(trainingData, 10)
    
    # Extract the X (feature data) and Y (label data)
    X_train, Y_train = preProcess.extractXY(train)
    X_test, Y_test = preProcess.extractXY(test)
    
    # Generate an instance for the Softmax Logisitc Regression Classifier with max iteration of 5000
    classifier = SoftmaxLogisticRegression(n_iter=5000)

    # Train the classifier. The training will stop once overfitting has been detected 
    # through cross-entropy
    classifier.fitOptimise(X_train, Y_train, X_test, Y_test)
    
    # Predict the test component of the training data
    Y_predict = classifier.predict(X_test)
    
    # Measure the accuracy
    report = classificationReport()
    print('Classifier Accuracy:', report.accuracyScore(Y_test, Y_predict))
    
    # Generate the Confusion Matrix
    print(report.confusionMatrix(Y_test, Y_predict))
    
    # Generate the Precision, Recall and F1-score metrics report
    report.metricsReport(Y_test, Y_predict)
    
    if crossvalidation:
        # Performance Cross Validation. The default value is 10.
        crossVal = CrossValidation()
        score = crossVal.crossValidationScore(classifier, X_input, Y_target, kfold)
        
        report = classificationReport()
        # Calculate the mean cross Validation Score
        print('Cross Validation Mean Score:', np.mean(score))
    
    # Predict the label for the test_data.csv
    toExport = np.empty((0,1))
    testingPredict = classifier.predict(testingData)
    
    # Predict the label for the test_data.csv
    for i in testingPredict:
        toExport = np.vstack((toExport,classLabelEncoder[i,-1:]))
    testingTitle = testing.readFile(input_path_data, testingDataFile)
    testingTitle = testingTitle.index.values[:, np.newaxis]

    # Export the predict label into the output folder
    toExport = np.hstack((testingTitle, toExport))
    
    testingLabelFile = os.path.join(path_data, os.path.join("output", testingLabelFile) )
    np.savetxt(testingLabelFile, toExport, delimiter=",",  fmt="%s")
    
    
