# Import Python Libraries
import pandas as pd
import datetime
import csv
import os
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.utils import check_array
from sklearn.utils.extmath import row_norms
from sklearn.utils.validation import FLOAT_DTYPES
import sklearn.utils.extmath  as ext



class DataReader:
    def __init__(self):
        # Default values for the Data Reader
        self.path = ''
        self.dataFile = None
        self.dataLabelFile = None
        self.labelDataFrame = np.empty((0,0))
    
    def readFile(self, path, file):
        # Combine the path and the file name
        file = os.path.join(path,file)
    
        # Read the csv data with column 0 as its index
        return pd.read_csv(file, sep=',', header=None, index_col=0)

    def columnEncoder(self, dataframe, column_number):
        classValues = {}
        
        # Get the unique sorted labels
        classes = dataframe.sort_values([column_number])[column_number].unique()
        
        # Assign a value for each label
        for i in range(len(classes)):
            classValues[i] = classes[i]

        self.labelDataFrame = pd.DataFrame([[key,value] for key,value in classValues.items()], index=classes)

        dataframe = pd.merge(dataframe, self.labelDataFrame[0].to_frame(0), how= 'outer', left_on=1, right_index= True)
        dataframe = dataframe.drop(dataframe.columns[0], axis=1)
        dataframe = dataframe.rename(columns={0: 1})
        
        return self.labelDataFrame.as_matrix(), dataframe

    def processData(self, path, dataFile, **kwargs):
        
        startDateTime = datetime.datetime.now()
        print('Start Reading Data:', startDateTime)
        
        # Variable Assignment
        self.path = path
        self.dataFile = dataFile
        
        # Read Optional Parameters
        self.dataLabelFile = kwargs.get('dataLabelFile')
        
        # Read the Data File
        print('Reading data file name:', self.dataFile)
        dataFrame = self.readFile(self.path, self.dataFile)
        
        if self.dataLabelFile is not None:
            # Read the Data Label
            print('Reading data label file name:', self.dataLabelFile)
            dataLabelFrame = self.readFile(self.path, self.dataLabelFile)
            
            print('Label encoding the class value...')
            outputColumnEncoder, dataLabelFrame = self.columnEncoder(dataLabelFrame, 1)
            
            numDataFrameColumn = len(dataFrame.columns)
            
            i = 1
            while i <= len(dataLabelFrame.columns):
                dataLabelFrame = dataLabelFrame.rename(columns={i: numDataFrameColumn + i})
                i+=1
            
            print('Combining data file and data label file...')
            dataFrame = dataFrame.join(dataLabelFrame, how='inner')
        
        matrixData = dataFrame.as_matrix()
        
        endDateTime = datetime.datetime.now()
        print('End Reading Training Data:', endDateTime)
        print('Elapsed Time:', endDateTime - startDateTime, '\n')

        if self.dataLabelFile is not None:
            return outputColumnEncoder, matrixData
        else:
            return matrixData

class PreProcessing:
    def __init__(self):
        # Default values for the Data Reader
        self.columnStart = 0        
        self.n = 500
    
    def removeZeroRow(self, data, columnStart, columnEnd):
        
        startDateTime = datetime.datetime.now()
        print('Start Reading Training Data:', startDateTime)
        
        print('Checking records with zero values across all dimension...')
        
        temp = data[:,columnStart:columnEnd].sum(axis=1)
        
        # Find rows with all of the feature values equal to zero
        rowIndex = np.where(data[:,columnStart:columnEnd].sum(axis=1)==0)
        
        print('Removing row numbers:', rowIndex[0])
        
        # Remove data with zero feature values
        data = np.delete(data,rowIndex,0)

        endDateTime = datetime.datetime.now()
        print('End Reading Training Data:', endDateTime)
        print('Elapsed Time:', endDateTime - startDateTime, '\n')
        
        return data

    def handleZerosInScale(self, scale, copy=True):
        # if we are fitting on 1D arrays, scale might be a scalar
        if np.isscalar(scale):
            if scale == .0:
                scale = 1.
            return scale
        elif isinstance(scale, np.ndarray):
            if copy:
                # New array to avoid side-effects
                scale = scale.copy()
            scale[scale == 0.0] = 1.0
            return scale

    def normalize(self, X):
        norm = 'l2'
        axis = 1
        copy = False
        sparse_format = 'csr'

        X = check_array(X, sparse_format, copy=copy, estimator='the normalize function', dtype=FLOAT_DTYPES)

        norms = row_norms(X)
        norms = self.handleZerosInScale(norms, copy=False)
        X /= norms[:, np.newaxis]

        return X

    def LSA(self, n):

        SVD = TruncatedSVD(n_components = n)
        #normalizer = Normalizer(copy=False)
    
        #return make_pipeline(SVD, normalizer)
        return SVD
    
    def LSAFitTransform(self, SVD, sourceData, skipLastColumn):

        startDateTime = datetime.datetime.now()
        print('Start Latent Semantic Analysis Fit Transform:', startDateTime)
        
        # Extract the last column if the skipLastColumn is set to True
        if skipLastColumn:
            data = SVD.fit_transform(sourceData[:, :-1])

            tempLastColumn = sourceData[:,-1:].ravel()
            
            data = self.normalize(data)
            
            data = np.column_stack((data,tempLastColumn))
        
        # Perform the TruncatedSVD.fit_transform method
        else:
            data = SVD.fit_transform(sourceData)
            data = self.normalize(data)    
        
        endDateTime = datetime.datetime.now()
        print('End Latent Semantic Analysis Fit Transform:', endDateTime)
        print('Elapsed Time:', endDateTime - startDateTime, '\n')
        
        return data

    def LSATransform(self, SVD, sourceData, skipLastColumn):

        startDateTime = datetime.datetime.now()
        print('Start Latent Semantic Analysis Transform:', startDateTime)
        
        if skipLastColumn is True:
            data = SVD.transform(sourceData[:, :-1])
            tempLastColumn = sourceData[:,-1:].ravel()
            data = np.column_stack((data,tempLastColumn))
        
        else:
            data = SVD.transform(data)
        
        data = self.normalize(data)
        
        endDateTime = datetime.datetime.now()
        print('End Latent Semantic Analysis Transform:', endDateTime)
        print('Elapsed Time:', endDateTime - startDateTime, '\n')
        
        return data

    def extractXY(self, data):
    
        return data[:, :-1], data[:,-1:].ravel()

    def trainTestSplit(self, data, testSize):
        
        # Extract the percentage
        if testSize % 10 != 0 and testSize != 0:
            print('Invalid Test Size! Setting the value to 10%.')
            numSplit = 1
        else:
            numSplit = testSize / 10
            
        uniqueClasses = len(np.unique(data[:, -1:]))

        dims = data.shape[1]

        dataSplit = np.empty((0,dims + 1))

        for c in range(uniqueClasses):
            rowIndex = np.where(data[:,-1:] == c)

            tempClassData = np.array_split(data[rowIndex[0],:], testSize) 

            for i in range(testSize):
                tempClassData[i] = np.insert(tempClassData[i], dims, i, axis=1)

                dataSplit = np.vstack((dataSplit, tempClassData[i]))

        return dataSplit[np.where(dataSplit[:,-1:] >= numSplit)[0],:-1], dataSplit[np.where(dataSplit[:,-1:] < numSplit)[0],:-1]

#Reads training data, checks rows if there are all zeros
#Finds out the top features for each category
#mainly use by Perceptron classifier    
class PerceptronDataReader:
    def __init__(self,n=0, trainFile="..\input\training_data.csv", trainLabelFile="..\input\training_labels.csv", testFile="..\input\test_data.csv", doCheck=False): 
        self.trainingDataFile = trainFile
        self.trainingLabelFile = trainLabelFile
        self.testFile = testFile
        
        
        #contain the titles and corresponding label
        self.titles = dict()
        self.doCheck = doCheck
        
        #each row is in the format: [title, [features], label]
        #to get title for a specific row: self.trainingData[row][0]
        #to get features for a specific row: self.trainingData[row][1]
        #features are array of strings, directly read from file
        #to get label for specific row: self.trainingData[row][2]
        self.trainingData = dict()
        
        #maximum no of features considering all labels
        #not  all features have value, what's the total features per label
        #the maximum total among all
        self.maxNumFeatures = 0
        
        
        #contains an array of row index per label
        #i.e.  {"Racing": [1,6,8,9..n]} means that self.trainingData[1] is a row belonging to label Racing
        self.trainingDataPerLabel = dict()

        #feature properties
        self.topFeaturesPerLabel = dict()
        self.topAllFeaturesPerLabel = dict()
        self.nonZeroColumnIndexPerLabel = dict()
        self.featureImportancePerLabel = dict()
        self.totalFeatures =   13626

        self.__topN = n

        #summaries
        self.labelList = None
        self.totalAppsPerLabel = dict()
        self.totalApps = 0
        
        
        #each row is in the format: [title,  [features]]
        #features are array of strings, directly read from file
        self.testData = []
        
        
        
    #reads the training file    
    def readTrainingFile(self):
        start = datetime.datetime.now()
        print("\nstart reading training file...")
        
        noValues = ["air.charcamera01", "com.andromo.dev72794.app76771","com.innoflame.employment_forum",
                    "com.gramedia.android.kidnesiap", "com.helloworld.hellomain",
                    "us.wmwm.bar", "air.SpeedCamIL",
                    "com.techrare.taxicalldriver"
            ]
        
        self.labelList = []
        dataTitles = []
        #get labels for each of the training data
        with open(self.trainingLabelFile, "rt", encoding="utf8") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                title = row[0]
                label = row[1]
                
                #During analysis, these titles were found to have no tf-idf values
                #will not include them in the training classification
                if (title in noValues): continue
                self.titles[title] = label
                if label not in self.labelList: self.labelList.append(label)
                dataTitles.append(title)
        
        """        
        usecols = (i for i in range(1,self.totalFeatures+1))
        csvdata = np.genfromtxt(trainData, dtype=np.float, usecols=usecols, delimiter=",")
        for index, title in dataTitles:

            #During analysis, these titles were found to have no tf-idf values
            #will not include them in the training classification
            if (title in noValues): continue
                
            features = csvdata[index]
            label = self.titles[title]

            self.trainingData[index] = [title, features, label, 0, 0]
            self.trainingDataPerLabel[label] = self.trainingDataPerLabel.get(label, [])

            #contains the index of the training data
            self.trainingDataPerLabel[label].append(index)
        """
            
        #read training data
        with open(self.trainingDataFile, "rt", encoding="utf8") as csvfile:
            reader = csv.reader(csvfile)
            index = 0
            for row in reader:
                colnum = len(row)
                features = row[1:colnum]
                
                title = row[0]
                
                if (index == 0):  self.totalFeatures = colnum - 1
                
                #During analysis, these titles were found to have no tf-idf values
                #will not include them in the training classification
                if (title in noValues): continue
                
                label = self.titles[title]
                self.trainingData[index] = [title, features, label, 0, 0]
                self.trainingDataPerLabel[label] = self.trainingDataPerLabel.get(label, [])

                #contains the index of the training data
                self.trainingDataPerLabel[label].append(index)
                
                index += 1
        

        print("total features", self.totalFeatures)
        print("\nLabels are", self.labelList)
        print("\n")                
        
        for k, rows in self.trainingDataPerLabel.items():
            totalRows = len(rows)
            print(k, "has total rows", totalRows)
            self.totalAppsPerLabel[k] = totalRows
            self.totalApps += totalRows
            
        end = datetime.datetime.now()
        print(index, "Training data loaded in ", end-start)
        
        
    #reads the test file
    def readTestFile(self):
        print("\nstart reading test file...")
        with open(self.testFile, "rt", encoding="utf8") as csvfile:
            reader = csv.reader(csvfile)
            index = 0
            for row in reader:
                colnum = len(row)
                features = row[1:colnum]
                title = row[0]

                self.testData.append([title, features])
                index += 1
        
        return len(self.testData) > 0
                

    def initTrainingData(self):
        self.readTrainingFile()
        
        start = datetime.datetime.now()
        for label in self.labelList:
            self.processByLabel(label)

        if self.doCheck:            
            print("check similiarity of features for each label")
            compared = []
            for label1 in self.labelList:
                for otherLabel in self.labelList:
                    if (otherLabel == label1 or (label1,otherLabel) in compared): continue
    
                    compared.append((label1,otherLabel))
                    compared.append((otherLabel,label1))
                    
                    lImpt = self.featureImportancePerLabel[label1]
                    oImpt = self.featureImportancePerLabel[otherLabel]
                    if (lImpt.sum() == oImpt.sum()):
                        print("WARNING:  same tf-idf values found for", label1, otherLabel, lImpt.sum(), oImpt.sum())
                            

        print("maximum total features per label:", self.maxNumFeatures)
        end = datetime.datetime.now()
        print("initializing data ended at", end-start)
        
        
        
    def processByLabel(self, label):
        
        print("selecting top features for label:", label)
        matrix = self.__getMatrixPerLabel(label)
        matrixByColumnSum = matrix.sum(axis=0)
        
        #normalize tfidf values by the no of documents in label
        matrixByColumnSum = matrixByColumnSum / len(self.trainingDataPerLabel[label])
        
        #find out the important features given tfidf
        sortIndex = np.argsort(matrixByColumnSum)
        nonzero = self.getNonZeroColumnIndex(label, matrix)
        nonzeroSorted = []
        for index in range(len(sortIndex)-1, -1, -1):
            if sortIndex[index] in nonzero:
                nonzeroSorted.append(sortIndex[index])
        
        self.maxNumFeatures = len(nonzeroSorted) if self.maxNumFeatures < len(nonzeroSorted) else self.maxNumFeatures
        
        self.topAllFeaturesPerLabel[label] = nonzeroSorted
        topFeatures = self.topAllFeaturesPerLabel[label][0:self.__topN] if (self.__topN > 0) else self.topAllFeaturesPerLabel[label]
        self.topFeaturesPerLabel[label] = sorted(topFeatures)
        matrixTop = matrix [:, topFeatures]
        self.featureImportancePerLabel[label] = np.mean(matrixTop, axis=0)

        matrixTopRowSum = matrixTop.sum(axis=1)
        
        if self.doCheck:        
            #check sum of each row if they have data,  need to increase N or Noise
            rowSumIndexes = np.where(matrixTopRowSum == 0)[0]
            if (rowSumIndexes.any()): 
                print("WARNING:   Found rows with all zero values: ", rowSumIndexes)
                for a in rowSumIndexes:
                    tdataRowId = self.trainingDataPerLabel[label][a]
                    matrixRow = matrix[a]
                    matrixRowNonZeros = np.nonzero(matrixRow)[0]
                    if (matrixRowNonZeros.any()):
                        print("\trow id:", tdataRowId, "matrix id:", a, "has non zero rows on:", np.nonzero(matrixRow)[0], "found at topFeatures index", [self.topAllFeaturesPerLabel[label].index(index) for index in matrixRowNonZeros])
                    else:
                        print("\t All zero values for", tdataRowId, self.getTitle(tdataRowId),"BOGGLING! ", matrixRow.sum(), matrixTopRowSum[a])

        
        #save tfidf for plotting
        matrixsum = matrix.sum(axis=1) / self.totalFeatures
        for i, rowIndex in enumerate(self.trainingDataPerLabel[label]):
            self.trainingData[rowIndex][3] = matrixsum[i]
                        
        #free-up space?
        del matrix
        del matrixTop
        
        
    #returns the row of features belonging to the given label
    #the returned format is a matrix of float arrays
    def __getMatrixPerLabel(self, label):
        
        rows = self.trainingDataPerLabel[label]
        
        data = []
        for rowIndex in rows:
            data.append([float(i) for i in self.trainingData[rowIndex][1]])
            #np.concatenate(data, self.trainingData[rowIndex][1])
                    
        return np.array(data, dtype=np.float)        
        #return data        
    
    #returns the title from training data or test data 
    def getTitle(self, index, isTraining=True):
        return self.trainingData[index][0] if isTraining else self.testData[index][0]
    
    #returns the label, note that test data has no label
    def getLabel(self, index):
        return self.trainingData[index][2]
    
    def getTopN(self): 
        if (self.__topN == 0): return self.maxNumFeatures
        else: return self.__topN
    
    def getFeatures(self, index):
        return self.trainingData[index][1]
        
    #returns the indexes of columns with values
    def getNonZeroColumnIndex(self, label, matrix=None):
        if (label not in self.nonZeroColumnIndexPerLabel):
            matrix = self.__getMatrixPerLabel(label) if matrix is None else matrix
            matrixSum = matrix.sum(axis=0)
            self.nonZeroColumnIndexPerLabel[label] = np.nonzero(matrixSum)[0]
            
        return self.nonZeroColumnIndexPerLabel[label]
    
    #removes zero columns from the matrix and returns a cleaner version
    def removeZeroColumns(self, matrix, nonzeroIndexes):
        return matrix[:,nonzeroIndexes]
    
    #return the average of tf-idf of each column in matrix
    def getFeatureImportance(self, label, matrix=None):
        
        if (label not in self.featureImportancePerLabel):
            matrix = matrix if matrix is not None else self.getMatrixTopFeatures(label)
            self.featureImportancePerLabel[label] = np.mean(matrix, axis=0)
            
        return self.featureImportancePerLabel[label]
    
    #return the column index with the top tf-idf     
    def getTopFeaturesPerLabel(self, label):
        if (label not in self.topFeaturesPerLabel):
            matrix = self.__getMatrixPerLabel(label)
            matrixSum = matrix.sum(axis=0)
            sortIndex = np.argsort(matrixSum)
            nonzero = self.getNonZeroColumnIndex(label, matrix)
            nonzeroSorted = []
            for index in range(len(sortIndex)-1, -1, -1):
                if sortIndex[index] in nonzero:
                    nonzeroSorted.append(sortIndex[index])
            
            self.topAllFeaturesPerLabel[label] = nonzeroSorted
            topFeatures = self.topAllFeaturesPerLabel[label][0:self.__topN] if (self.__topN > 0) else self.topAllFeaturesPerLabel[label]
            self.topFeaturesPerLabel[label] = sorted(topFeatures)

        else: return self.topFeaturesPerLabel[label]
    
    #returns the matrix with top tf-idf columns
    def getMatrixTopFeatures(self, label):
        topfeatures = self.topAllFeaturesPerLabel(label)[0:self.__topN] 
        return self.__getMatrixPerLabel(label)[:, topfeatures]
    
    
    #returns the topN features for specific row index
    #called from getRowFeturesAsArray
    def __getTopRowFeaturesAsArray(self, index, label=None):
        label = self.getLabel(index) if label is None else label
        
        #get the column index with the top tf-idf
        topFeatures = self.getTopFeaturesPerLabel(label)
        features = np.array([float(val) for i, val in enumerate(self.getFeatures(index)) if i in topFeatures ], dtype=np.float)
        return features
        
    #returns the features for a specific row index
    #the returned format is a vector of float numbers
    #if n is given, then will return the topN features
    def getRowFeaturesAsArray(self, index, label=None):
        if (self.__topN == 0):
            return np.array([float(i) for i in self.getFeatures(index)], dtype=np.float)
        else:
            return self.__getTopRowFeaturesAsArray(index, label)
        

    def getTestRowFeaturesAsArray(self, index, label=None):
        if (len(self.testData) == 0): self.readTestFile()
        if (label is None):
            return np.array([float(i) for i in self.testData[index][1]], dtype=np.float)
        else:
            topFeatures = self.getTopFeaturesPerLabel[label]
            features = np.array([float(val) for i, val in enumerate(self.testData[index][1]) if i in topFeatures ], dtype=np.float)
            return features
    
    def getTestRowTitle(self, index):
        return self.testData[index][0]      

#Abstract class to massage data returned from PerceptronDataReader
#in preparation for perceptron classifier.    
class InputParser:
    
    def __init__(self, reader):
        self.labelList = reader.labelList
        self.labelIndex = dict()
        for i, label in enumerate(reader.labelList):
            self.labelIndex[label] = i
        self.rows = []
        self.titles = []
        self.labels = []
        self.topN = reader.getTopN()
        self.testRows = []    
        
        for index in range(reader.totalApps):
            self.titles.append(reader.getTitle(index))
            self.labels.append(reader.getLabel(index))
            

    def getTitle(self, index): return self.titles[index]    
    def getData(self): return self.rows
    
    def getTestData(self): return self.testRows
    
    def getRow(self, index, label): pass
        
    def getLabels(self):
        return self.labelList
    
    #Returns the label for the data row    
    def getLabel(self, index):
        return self.labels[index]
    
    #Returns the index no of the label
    def getLabelIndex(self, label): return self.labelIndex[label]
    
    #Returns the default weight to be use by perceptron
    #Sets to 1 for bias
    def getDefaultWeights(self): return 1


#A subclass of InputParser to read data from PerceptronDataReader
#and generates features for each category using the top features
class TFIDFInputParser(InputParser):
    def __init__(self, reader):
        InputParser.__init__(self, reader)

        self.totalAppsPerLabel = reader.totalAppsPerLabel
        self.totalApps = reader.totalApps
        self.testTitles = []
        
        for index in range(reader.totalApps):
            arr = np.array([float(i) for i in reader.getFeatures(index)], dtype=np.float)
            row = []
            for i, label in enumerate(self.getLabels()):
                topFeatures = reader.getTopFeaturesPerLabel(label)
                row.append(arr[topFeatures].sum() / self.topN)
                
            self.rows.append(row)
            
            #get a peak
            if (index % 5000 == 0): print(index,self.getLabelIndex(self.getLabel(index)), row)
            
        if (len(reader.testData) > 0):
            for index in range(len(reader.testData)):
                arr = np.array([float(i) for i in reader.testData[index][1]], dtype=np.float)
                row = []
                for i, label in enumerate(self.getLabels()):
                    topFeatures = reader.getTopFeaturesPerLabel(label)
                    row.append(arr[topFeatures].sum() / self.topN)
                    
                self.testRows.append(row)
                self.testTitles.append(reader.testData[index][0])
                
        
    def getTestData(self): return self.testRows
    
            
    def getRow(self, index, label, test=None):
        rows = self.testRows if test else self.rows
        tfidf = rows[index][self.getLabelIndex(label)]
        #returns the probability (naive bayes) of the data to
        #be of type label
        tfidf =  tfidf * self.calcPApp(label) 
        return tfidf

    def calcPApp(self, label):
        totalLabel = self.totalAppsPerLabel[label]
        return totalLabel / self.totalApps


class BayesTraining:
    CATEGORY = 30
    KMAX = 0
    
    TotalCount = 0
    TotalifdfValinCat = np.zeros(CATEGORY)
    TotalCountCat = np.zeros(CATEGORY)
    

    def __init__(self, K):
        #self.trainmatrix = trainmatrix
        self.KMAX =  K
        self.prior = np.zeros(self.CATEGORY)
        self.WordLikelihood = np.zeros((self.CATEGORY, self.KMAX)) 
        return
        
        
    def ProcessData(self,X, Y):
          print("calculating prior and likelihood")
          df = pd.DataFrame(X)
          df['labels'] = Y
          grouped = df.groupby(['labels'])
          for cat,group in grouped:
                df1 = pd.DataFrame(group)
                newdf = df1.drop(df1.columns[[-1]], axis= 1,inplace = False)
                groupcat =     df1.iloc[0]['labels']
                self.TotalCount += len(group)
                self.TotalCountCat[int(groupcat)] = len(group)
    
    
                self.TotalifdfValinCat[int(groupcat)] = newdf.values.sum()

                for col in newdf:
                    self.WordLikelihood[int(groupcat)][int(col)] = newdf[int(col)].sum()
            
    
          self.TotalCount = np.sum(self.TotalCountCat)
          self.prior = self.TotalCountCat / self.TotalCount
          print("sum of priors", np.sum(self.prior))
          #print("sum likelihood", np.sum(self.WordLikelihood))
    
          return
    
   
    #Calculate likelihood of each word
    def CalculateAlllikelihood(self):
        
        for i in range(self.CATEGORY):
            #calculate word likelihood
            for col in range(self.KMAX): 
               
               self.WordLikelihood[i][col] =  (1 + self.WordLikelihood[i][col]) / (self.KMAX + np.sum(self.TotalifdfValinCat[i]))
               
               
            
        #print(self.TotalifdfValinCat)

        print("Sum of all likelihoods in catgories combined",np.sum(self.WordLikelihood) )

        return
            
        

class FeatureSelection:
   
        
    def FitTransformBayes(self,X,Y):
          startDateTime = datetime.datetime.now()

          print("Start Feature Selection....", startDateTime)
    
          
          df = pd.DataFrame(X)
          WordImpPercent = np.zeros((30, df.shape[1]))
          df['labels'] = Y
          grouped = df.groupby(['labels'])
          self.idx = []
          print("Calculate weights of column within categories")

          for cat,group in grouped:
                df1 = pd.DataFrame(group)
                newdf = df1.drop(df1.columns[[-1]], axis= 1,inplace = False)
                groupcat =     df1.iloc[0]['labels']    
                sum = newdf.values.sum()
                
                #Get word weight in a column
                for col in newdf:
                    WordImpPercent[int(groupcat)][int(col)] =  (newdf[int(col)].sum()/ sum) *100
                    if (WordImpPercent[int(groupcat)][int(col)] > 0.055):
                       self.idx.append(col)
          
          
          #df1 = pd.DataFrame(WordImpPercent)
          #sum = df1.values.sum()
          #print("Calculate weight,variance across categories")
          #idx = []
          #for col in df1:
          #     catcol = df1[col].sum()
          #     var = df1[col].var()
          #     if ( (catcol/sum) * 100  <= 0.01   and var >= 0):
          #        self.idx.append(col)
          #     elif(catcol/sum) * 100  > 0.01:
          #        self.idx.append(col)

          #store selected column for future test transform
          self.idx = list(set(self.idx))
          print("Selected {} columns".format(len(self.idx)))
          
          #Transform data
          newdf = df[self.idx]
          endDateTime = datetime.datetime.now()
          print('End Feature Selection Transform:', endDateTime)
          return (newdf.as_matrix())
          
    def Transform(self,X, ModelNo):
           #Transform test data
         if ModelNo == 1:
           print('Transform Test data...')
           df = pd.DataFrame(X)
           df = df[self.idx]
           X = df.as_matrix()
         elif ModelNo == 2:
           X = ext.safe_sparse_dot(X,self.V)
           
           return X

    def FitTransformKnn(self, X,n_components):
          startDateTime = datetime.datetime.now()

          print('Start Feature Selection using svd ',startDateTime)
          U, s, Vt = ext.randomized_svd(X, n_components)
          self.V = Vt.T
          #U,s, Vt =  sparsesvd.svds(X,n_components)
          S = np.diag(s)
          X = np.dot(U, S)
          
          endDateTime = datetime.datetime.now()

          print('End Feature Selection ',endDateTime)
          return X
          
          
    def FeatureSelection(self, Train, ModelNo):
       tempLastColumn = Train[:,-1:].ravel()
       if  ModelNo == 1:
         trainingData = self.FitTransformBayes(Train[:, :-1],tempLastColumn)
       elif ModelNo == 2 or ModelNo == 4:
         #295
         trainingData = self.FitTransformKnn(Train[:, :-1],280)
       trainingData = np.column_stack((trainingData,tempLastColumn))
       return trainingData

        