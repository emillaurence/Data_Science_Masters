# Import Python Libraries
import pandas as pd
import datetime
import numpy as np
import pickle
from DataLibrary import BayesTraining
from sklearn.metrics.pairwise import cosine_similarity


class SoftmaxLogisticRegression:
    def __init__(self, **kwargs):
        self.W = np.empty((0,0))
        self.b = np.empty((0,0))
        self.n_iter = 1500
        self.learning_rate = 0.001
        
        if kwargs.get('n_iter') is not None:
            self.n_iter = kwargs.get('n_iter')
        if kwargs.get('learning_rate') is not None:
            self.learning_rate = kwargs.get('learning_rate')
        if kwargs.get('learning_rate') is not None:
            self.learning_rate = kwargs.get('learning_rate')
        
    def yIndicator(self, y, K):
        N = len(y)
        ind = np.zeros((N, K))
        for i in range(N):
            ind[i, int(y[i])] = int(1)
        return ind

    def softmax(self, a):
        expA = np.exp(a)
        return expA / expA.sum(axis=1, keepdims=True)

    def forward(self, X, W, b):
        return self.softmax(X.dot(W) + b)
    
    def predicts(self, pYtrain):
        return np.argmax(pYtrain, axis=1)
    
    def predict(self, X_test):
        return np.argmax(self.forward(X_test, self.W, self.b), axis=1)

    def crossEntropy(self, T, pY):
        return -np.mean(T*np.log(pY))

    def classificationRate(self, Y, P):
        return np.mean(Y == P)

    def fit(self, X_train, Y_train):
        
        startDateTime = datetime.datetime.now()
        print('Start Training Data:', startDateTime)
        
        self.W = np.empty((0,0))
        self.b = np.empty((0,0))
        
        D = X_train.shape[1]
        K = len(set(Y_train))
        Ytrain_ind = self.yIndicator(Y_train, K)

        # Randomly Initialise Weights
        self.W = 0.001*np.random.randn(D, K)
        self.b = np.zeros(K)
        
        train_costs = []
        test_costs = []
        
        for i in range(self.n_iter):
            pYtrain = self.forward(X_train, self.W, self.b)
            #pYtest = self.forward(X_test, self.W, self.b)

            ctrain = self.crossEntropy(Ytrain_ind, pYtrain)

            # Gradient Descent
            self.W -= self.learning_rate*(X_train.T.dot(pYtrain - Ytrain_ind))
            self.b -= self.learning_rate*(pYtrain - Ytrain_ind).sum(axis=0)
            #if i % 100 == 0:
            #    print(i, self.classificationRate(Y_train, self.predicts(pYtrain)), self.classificationRate(Y_test, self.predicts(pYtest)))
        
        endDateTime = datetime.datetime.now()
        print('End Training Data:', endDateTime)
        print('Elapsed Time:', endDateTime - startDateTime, '\n')
		
    def fitOptimise(self, X_train, Y_train, X_test, Y_test):
        
        startDateTime = datetime.datetime.now()
        print('Start Training Data:', startDateTime)
        
        self.W = np.empty((0,0))
        self.b = np.empty((0,0))
        
        previous_ctest = 1
		
        D = X_train.shape[1]
        K = len(set(Y_train))
        Ytrain_ind = self.yIndicator(Y_train, K)
        Ytest_ind = self.yIndicator(Y_test, K)

        # Randomly Initialise Weights
        self.W = 0.001*np.random.randn(D, K)
        self.b = np.zeros(K)
        
        train_costs = []
        test_costs = []
        
        for i in range(self.n_iter):
            pYtrain = self.forward(X_train, self.W, self.b)
            pYtest = self.forward(X_test, self.W, self.b)

            ctrain = self.crossEntropy(Ytrain_ind, pYtrain)
            ctest = self.crossEntropy(Ytest_ind, pYtest)

            # Gradient Descent
            self.W -= self.learning_rate*(X_train.T.dot(pYtrain - Ytrain_ind))
            self.b -= self.learning_rate*(pYtrain - Ytrain_ind).sum(axis=0)
            #if i % 100 == 0:
            #    print(i, ctrain, ctest, self.classificationRate(Y_train, self.predicts(pYtrain)), self.classificationRate(Y_test, self.predicts(pYtest)))
            if previous_ctest < ctest:
                break
            
            previous_ctest = ctest

        endDateTime = datetime.datetime.now()
        print('End Training Data:', endDateTime)
        print('Elapsed Time:', endDateTime - startDateTime, '\n')

#Implements the Perceptron Classifier, an online learning
#algorithm.  This implementation is for multi-class classification        
class PerceptronClassifier():
    
    def __init__(self, nEpoch, minput):
        self.nEpoch = nEpoch
        self.input = minput
        self.errors = []
    
    #updates the weight if predicted and expected are not the same
    #since we add 1 bias, we subtract the loss/error from the expected to move it closer to 0
    #and we add the loss to predicted
    def __calcWeight(self, weights, predictedLabel, expectedLabel, index):
        weightsNext = [weight for weight in weights]
        
        if (predictedLabel != expectedLabel):
        
            labelIndex = self.input.getLabelIndex(expectedLabel)
            expectedCurrWeights = weights[labelIndex]
            weightsNext[labelIndex] =  expectedCurrWeights - self.getError(predictedLabel, expectedLabel, index)
            
            predIndex = self.input.getLabelIndex(predictedLabel)
            predCurrWeights = weights[predIndex]
            weightsNext[predIndex] =  predCurrWeights + self.getError(predictedLabel, expectedLabel, index)
            
        return weightsNext

    #separated for plotting loss function        
    def getError(self, predictedLabel, expectedLabel, index):
        return self.input.getRow(index, expectedLabel) - self.input.getRow(index, predictedLabel)
        
    
    #predicts using dot function of weights and features    
    def __predict(self, index, weights):
        
        predicted_score = 0
        predicted_label = self.input.labelList[0]

        for i, label in enumerate(self.input.getLabels()):
            val = self.input.getRow(index, label) 
            labelIndex = self.input.getLabelIndex(label)
            weight = weights[labelIndex] 
            score = np.dot(val, weight)

            if score >= predicted_score:
                predicted_label = label
                predicted_score = score
            
        return predicted_label
    
    #train the classifier by updating the weights if wrong prediction is made
    #learning iterations is set by epoch parameter
    def __perceptronTrain(self, trainingSet, weightsb4 = None):
        weights = weightsb4 if weightsb4 else [[self.input.getDefaultWeights() for label in self.input.getLabels()]]
        ave = dict()
        for i, w in enumerate(weights[0]):  ave[i] = w
        for n in range(self.nEpoch):
            total_error = 0
            for index in trainingSet:
                expectedLabel = self.input.getLabel(index)
                weightsCurr = weights[-1]
                predictedLabel = self.__predict(index, weightsCurr)
                weightsNext = self.__calcWeight(weightsCurr, predictedLabel, expectedLabel, index)
                total_error += self.getError(predictedLabel, expectedLabel, index)
                weights.append(weightsNext)
                
                for i, w in enumerate(weightsNext): ave[i] += w
            
            #just for plotting
            if len(self.errors) < self.nEpoch: self.errors.append(total_error)   

        average_weights = []
        for k, w in ave.items():
            average_weights.append(w/len(weights))
            
        return average_weights
    
    
    def getErrors(self):
        return self.errors
        
    #main method called to classify test by learning train data           
    def classify(self, train, test):
        predictions = []
        weights = self.__perceptronTrain(train, None)
        for index in test:
            predictedLabel = self.__predict(index, weights)
            predictions.append(predictedLabel)
            
        self.saved_weights = weights
        return predictions

    #this is just a helper method to save weights for offline
    #not used
    def dumpWeights(self, weights):
        print("Dumping weights to perceptron_weights.pkl")
        with open(self.weightsFile, "wb") as file:
            pickle.dump(weights, file )
            

class BayesNaiveClassifier(BayesTraining):
    
    def __init__(self, ColCount):
        super().__init__(ColCount)
        self.PosteriorProbability = np.zeros(self.CATEGORY)
        self.probabilityinC = np.zeros(self.CATEGORY)
        
        return

     
    def fit(self,Xtrain, Ytrain):
        print("Fitting data to Naive Bayes")
        self.ProcessData(Xtrain, Ytrain)
        self.CalculateAlllikelihood()
        return
            
    
    def PolynomialProbability(self, UnclassifiedDoc):
        #probabilityinC = np.zeros(self.CATEGORY)
        #for wordifdf in UnclassifiedDoc:
        for row in range(0,self.CATEGORY): 
            self.probabilityinC[row] = 1
            #for col in range(0, self.KMAX):
               #if self.WordLikelihood[row][col] <= 0:
                    #print("oops",row,col)
               #if UnclassifiedDoc[col] <= 0:
                      #UnclassifiedDoc[col] = 0

               
              # self.probabilityinC[row] = self.probabilityinC[row] * \
              #                                   (self.WordLikelihood[row][col] ** UnclassifiedDoc[col])
            self.probabilityinC[row] = np.prod( np.power( self.WordLikelihood[row] ,UnclassifiedDoc))
        #print(self.probabilityinC)
        return 
    

    #input is k attributes of numerical data 
    def predict(self, UnclassifiedDoc):
        print("Start prediction with Naive Bayes")

        Y_predict = []
        for doc in UnclassifiedDoc:
           self.PolynomialProbability(doc)
           #print(self.probabilityinC)

           self.PosteriorProbability = self.probabilityinC * self.prior
           Y_predict.append(np.argmax(self.PosteriorProbability))
           
        return Y_predict

class knn:
    
     def __init__(self, k):
        self.k = 30
        self.KNN = k
    
     def fit(self, X_Test, X_Labels):
        self.X = X_Test
        self.XLabels = X_Labels

         
     def CountVotes(self, kNearest):
        count = np.zeros(self.k)
        for i in range(self.KNN):
            label = self.XLabels[kNearest[i]]
            count[int(label)] += 1
        winner = np.argmax(count) 
        #print ("Winner {}".format(winner))
        return winner

     def predict(self, Y_Test):
        YPredition = []
        #Calculate cosine similarity on whole data set
        #sort in descending order select based on vote
        print('Calculating cosine similarity')
        dst = cosine_similarity(Y_Test, self.X)
        print('Voting K most similar')

        for i in range (len(dst)):
            order = np.argsort(dst[i])[::-1]
            kNearest = order[:self.KNN]
            #print(Y_Test[i])
            #print(dst[i], order, kNearest)
            label = self.CountVotes(kNearest)
            #print(label)
            YPredition.append(label)
        return YPredition

        
 
            
