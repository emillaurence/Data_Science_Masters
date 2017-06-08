# Import Python Libraries
import pandas as pd
import datetime
import numpy as np
from random import seed
from random import randrange


class CrossValidation:
    def __init__(self, **kwargs):
        self.k_fold = 10
        self.testSize = 10
        
    def accuracyScore(self, Y_test, Y_predict):
        
        actual = 0
        
        expected = len(Y_test)
        
        for n in range(expected):
            if Y_test[n] == Y_predict[n]:
                actual+=1
        
        return actual/expected
    
    def crossValidationScore(self, classifier, X, Y, k_fold, **kwargs):
        
        startDateTime = datetime.datetime.now()
        print('Start Cross Validation:', startDateTime)
        
        if kwargs.get('testSize') is not None:
            self.testSize = kwargs.get('testSize')

        self.k_fold = k_fold
        
        Y = np.array(Y)[:, np.newaxis]

        score = []

        uniqueClasses = len(np.unique(Y))

        X_dims = X.shape[1]
        Y_dims = Y.shape[1]
        
        X_CV = np.empty((0,X_dims + 1))
        Y_CV = np.empty((0,Y_dims + 1))
        
        print('Assigning fold number to the data...')
        for c in range(uniqueClasses):
            rowIndex = np.where(Y[:,-1:] == c)

            tempXClassData = np.array_split(X[rowIndex[0],:], self.testSize) 
            tempYClassData = np.array_split(Y[rowIndex[0],:], self.testSize) 
            
            for i in range(self.testSize):

                tempXClassData[i] = np.insert(tempXClassData[i], X_dims, i, axis=1)
                X_CV = np.vstack((X_CV, tempXClassData[i]))

                tempYClassData[i] = np.insert(tempYClassData[i], Y_dims, i, axis=1)
                Y_CV = np.vstack((Y_CV, tempYClassData[i]))

        for fold in range(self.k_fold):
            print('Processing fold number:', fold + 1, '\n')
            rowIndex = np.where(Y_CV[:,-1:] == fold)
            X_test = X_CV[rowIndex[0], :-1]
            Y_test = Y_CV[rowIndex[0], :-1].ravel().astype('int')

            rowIndex = np.where(Y_CV[:,-1:] != fold)
            X_train = X_CV[rowIndex[0], :-1]
            Y_train = Y_CV[rowIndex[0], :-1].ravel().astype('int')

            classifier.fitOptimise(X_train, Y_train, X_test, Y_test)

            Y_predict = classifier.predict(X_test)

            score.append(self.accuracyScore(Y_test, Y_predict))

        endDateTime = datetime.datetime.now()
        print('End Cross Validation:', endDateTime)
        print('Elapsed Time:', endDateTime - startDateTime, '\n')
        
        return score

#Splits the data to N folds, then take 1 fold for test data
#This is repeated by N times, taking a different set of fold for test data
#same implementation with scipy KFold
#nFolds must be atleat 2
class MyKFold:
    def getFolds(self, data, nFolds, random = True):
        
        if (nFolds < 2): 
            print("The minimum nFold should be 2!")
            raise
        
        trainData = []
        testData = []
        seed(1)
        
        total = len(data)
        testSize = int(total / nFolds)
        
        remaining = [j for j in range(total - (testSize*nFolds))]
        for i in range(nFolds):
            train = []
            test = []
            
            start = i * testSize
            end = start + testSize
            dataCopy = [j for j in range(total)]
            ctr = 0
            isPopped = False
            while len(dataCopy) > 0:
               
                if ctr >= start and ctr < end:
                    if (random): testIndex = randrange(len(dataCopy))
                    else: testIndex = 0
                    test.append(dataCopy.pop(testIndex))
                    
                    if (isPopped == False and len(remaining) > 0): 
                        
                        if (random): testIndex =  randrange(len(dataCopy))
                        else: testIndex = 0
                        
                        test.append(dataCopy.pop(testIndex))
                        remaining.pop(0)
                        isPopped = True
                else:
                    if (random):  trainIndex = randrange(len(dataCopy)) 
                    else: trainIndex = 0
                    train.append(dataCopy.pop(trainIndex))
                
                ctr += 1
                
            trainData.append(train)
            testData.append(test)
            
        return trainData, testData
    
