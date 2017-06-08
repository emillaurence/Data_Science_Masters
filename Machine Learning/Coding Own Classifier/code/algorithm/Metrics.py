# Import Python Libraries
import pandas as pd
import datetime
import numpy as np

class classificationReport:
    def accuracyScore(self, Y_test, Y_predict):
        
        actual = 0
        
        expected = len(Y_test)
        
        for n in range(expected):
            if Y_test[n] == Y_predict[n]:
                actual+=1
        
        return actual/expected

    def confusionMatrix(self, Y_test, Y_predict):
        
        size = len(np.unique(Y_predict))

        confusionMatrix = np.zeros((size, size)).astype('int')

        for i in range(len(Y_predict)):

            if Y_test[i] == Y_predict[i]:
                confusionMatrix[Y_predict[i],Y_predict[i]]+= int(1)
                
            else:
                confusionMatrix[int(Y_test[i]),Y_predict[i]]+= int(1)
        
        return confusionMatrix
    
    def metricsReport(self, Y_test, Y_predict):
        
        size = len(np.unique(Y_predict))
        
        confusion = self.confusionMatrix(Y_test, Y_predict)

        print('Class'.rjust(15), 'Precision'.rjust(15), 'Recall'.rjust(15), 'f1-score'.rjust(15), 'Support'.rjust(15))

        support = []
        recall = []
        precision = []
        f1 = []

        for i in range(size):
            # Support
            support.append(confusion[i, :].sum())

            # Recall
            recall.append(confusion[i, i]/ confusion[i, :].sum())

            # Precision
            precision.append(confusion[i, i]/ confusion[:, i].sum())

            # f1-score
            f1.append(2 * (precision[i] * recall[i]) / (precision[i] + recall[i]))

            print(str('Class ' + str(format(i, '02d'))).rjust(15), str(format(precision[i], '.2f')).rjust(15), str(format(recall[i], '.2f')).rjust(15), str(format(f1[i], '.2f')).rjust(15), str(support[i]).rjust(15))

        print('Avg/ Total'.rjust(15), format(np.mean(precision), '.2f').rjust(15), format(np.mean(recall), '.2f').rjust(15), format(np.mean(f1), '.2f').rjust(15), str(np.sum(support)).rjust(15))

#Calculates the recall. precision, f-score and their 
#corresponding macro and micro averages
class Scores:
    
    def __init__(self): pass
        
    def getScores(self, labelList, y_test, y_predict):
        #tp, fp, fn, tn
        metrics = []
        
        confusion = self.getConfusionNumpy(labelList, y_test, y_predict)
        predicted_positive = np.sum(confusion, axis=0)
        actual_positive = np.sum(confusion, axis=1)
        total = np.sum(predicted_positive)
        total_precision = 0
        total_recall = 0
        total_fscore = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_labels = len(labelList)
        for i, actual_label in enumerate(labelList):
            for j , predicted_label in enumerate(labelList):
                if (i != j): continue
                tp = confusion[i,j]
                fp = predicted_positive[j] - tp
                fn = actual_positive[j] - tp
                tn = total - tp -fp - fn
                s = dict()
                s["label"] = predicted_label
                s["tp"] = tp
                s["fp"] = fp
                s["fn"] = fn
                s["tn"] = tn
                s["precision"] = float(tp /(tp+fp)) if (tp > 0) else 0
                s["recall"] = float( tp / (tp + fn) ) if (tp > 0) else 0
                s["f-score"] = (2 * s["precision"] * s["recall"]) / (s["precision"] + s["recall"]) if tp > 0 else 0
                
                total_precision += s["precision"]
                total_recall += s["recall"]
                total_fscore += s["f-score"]
                total_tp += tp
                total_fp += fp
                total_fn += fn
                
                metrics.append(s)
                print(predicted_label, "metrics:", s)
        
        #calculate macro and micro averge
        macro_precision = total_precision / total_labels
        macro_recall = total_recall / total_labels
        macro_fscore = total_fscore / total_labels
        
        micro_precision = total_tp / (total_tp + total_fp)
        micro_recall = total_tp / (total_tp + total_fn)
        #micro_fscore = micro_precision + micro_recall  / 2
        micro_fscore = (2 * micro_precision * micro_recall) /(micro_precision + micro_recall)
        
        print("total TP:", total_tp, "total FP:", total_fp, "total FN:", total_fn)
        print("total labels:", total_labels)
        print("macro_precision:", macro_precision,"macro_recall", macro_recall, "macro_fscore", macro_fscore)
        print("micro_precision", micro_precision,"micro_recall", micro_recall,"micro_fscore", micro_fscore)
        
        overall = dict()
        overall["macro_precision"] = macro_precision
        overall["macro_recall"] = macro_recall
        overall["macro_fscore"] = macro_fscore
        overall["micro_precision"] = micro_precision
        overall["micro_recall"] = micro_recall
        overall["micro_fscore"] = micro_fscore
        overall["metrics"] = metrics
        
        return overall
    
    def getConfusionNumpy(self, labelList, y_test, y_predict):
        lenC = len(labelList)
        confusion = np.zeros(shape=(lenC, lenC))
        for i, actual in enumerate(y_test):
            predict = y_predict[i]
            actualI = labelList.index(actual)
            predictI = labelList.index(predict)
            confusion[actualI][predictI] += 1
        return confusion
        
    

    