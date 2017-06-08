import numpy as np
import pandas as pd
import datetime
import os

from optparse import OptionParser
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef
from imblearn.metrics import classification_report_imbalanced
from sklearn.svm import LinearSVC


def getConnection():
    import psycopg2
    
    try: 
        conn = psycopg2.connect(database='COMP5318', host='comp5318.cge7dneddtek.ap-southeast-2.rds.amazonaws.com',
                                user='COMP5318', password='COMP5318')
        print('AWS PostgreSQL DB Connection Successful!')
        return conn
    except Exception as e:
        print("Unable to connect to the database")
        print(e)

#Query Function
def pgquery( conn, sqlcmd, args ):
   """ utility function to execute some SQL query statement
       can take optional arguments to fill in (dictionary)
       will print out on screen the result set of the query
       error and transaction handling built-in """
   retval = False
   query_result = []
   with conn:
      with conn.cursor() as cur:
         try:
            if args is None:
                cur.execute(sqlcmd)
            else:
                cur.execute(sqlcmd, args)
            for record in cur:
                query_result.append(record)
            retval = True
         except Exception as e:
            print("DB Read Error: ")
            print(e)
   return query_result

#Execution Function
def pgexec( conn, sqlcmd, args, msg ):
   """ utility function to execute some SQL statement
       can take optional arguments to fill in (dictionary)
       error and transaction handling built-in """
   retval = False
   with conn:
      with conn.cursor() as cur:
         try:
            if args is None:
               cur.execute(sqlcmd)
            else:
               cur.execute(sqlcmd, args)
            if msg is not None:
                print("Success: " + msg)
            retval = True
         except Exception as e:
            print("DB Error: ")
            print(e)
   return retval


def getTrainDataframe():
    print("Retrieving training data")
    query_stmt ="""
    SELECT
     cast(age AS INTEGER)
    ,CASE WHEN TRIM(workclass)='Federal-gov' THEN 1 ELSE 0 END AS  Federal_gov
    ,CASE WHEN TRIM(workclass)='Local-gov' THEN 1 ELSE 0 END AS  Local_gov
    --,CASE WHEN TRIM(workclass)='Never-worked' THEN 1 ELSE 0 END AS  Never_worked
    ,CASE WHEN TRIM(workclass)='Private' THEN 1 ELSE 0 END AS  Private
    ,CASE WHEN TRIM(workclass)='Self-emp-inc' THEN 1 ELSE 0 END AS  Self_emp_inc
    ,CASE WHEN TRIM(workclass)='Self-emp-not-inc' THEN 1 ELSE 0 END AS  Self_emp_not_inc
    ,CASE WHEN TRIM(workclass)='State-gov' THEN 1 ELSE 0 END AS  State_gov
    ,CASE WHEN TRIM(workclass)='Without-pay' THEN 1 ELSE 0 END AS  Without_pay
    ,cast(fnlwgt AS INTEGER)
    ,CAST(education_num AS INTEGER)
    ,CASE WHEN TRIM(marital_status) IN ('Married-AF-spouse', 'Married-civ-spouse') THEN 1
          WHEN TRIM(relationship) IN ('Husband', 'Wife') THEN 1
          ELSE 0 END AS  Married
    ,CASE WHEN TRIM(occupation)	='Adm-clerical' THEN 1 ELSE 0 END AS  Adm_clerical
    ,CASE WHEN TRIM(occupation)	='Armed-Forces' THEN 1 ELSE 0 END AS  Armed_Forces
    ,CASE WHEN TRIM(occupation)	='Craft-repair' THEN 1 ELSE 0 END AS  Craft_repair
    ,CASE WHEN TRIM(occupation)	='Exec-managerial' THEN 1 ELSE 0 END AS  Exec_managerial
    ,CASE WHEN TRIM(occupation)	='Farming-fishing' THEN 1 ELSE 0 END AS  Farming_fishing
    ,CASE WHEN TRIM(occupation)	='Handlers-cleaners' THEN 1 ELSE 0 END AS  Handlers_cleaners
    ,CASE WHEN TRIM(occupation)	='Machine-op-inspct' THEN 1 ELSE 0 END AS  Machine_op_inspct
    ,CASE WHEN TRIM(occupation)	='Other-service' THEN 1 ELSE 0 END AS  Other_service
    ,CASE WHEN TRIM(occupation)	='Priv-house-serv' THEN 1 ELSE 0 END AS  Priv_house_serv
    ,CASE WHEN TRIM(occupation)	='Prof-specialty' THEN 1 ELSE 0 END AS  Prof_specialty
    ,CASE WHEN TRIM(occupation)	='Protective-serv' THEN 1 ELSE 0 END AS  Protective_serv
    ,CASE WHEN TRIM(occupation)	='Sales' THEN 1 ELSE 0 END AS  Sales
    ,CASE WHEN TRIM(occupation)	='Tech-support' THEN 1 ELSE 0 END AS  Tech_support
    ,CASE WHEN TRIM(occupation)	='Transport-moving' THEN 1 ELSE 0 END AS  Transport_moving
    ,CASE WHEN TRIM(race)	='Amer-Indian-Eskimo' THEN 1 ELSE 0 END AS  Amer_Indian_Eskimo
    ,CASE WHEN TRIM(race)	='Asian-Pac-Islander' THEN 1 ELSE 0 END AS  Asian_Pac_Islander
    ,CASE WHEN TRIM(race)	='Black' THEN 1 ELSE 0 END AS  Black
    ,CASE WHEN TRIM(race)	='Other' THEN 1 ELSE 0 END AS  Other
    ,CASE WHEN TRIM(race)	='White' THEN 1 ELSE 0 END AS  White
    ,CASE WHEN TRIM(sex)	='Female' THEN 1 ELSE 0 END AS  Gender
    ,CAST(capital_gain AS INTEGER) AS capital_gain
    ,CAST(capital_loss AS INTEGER) AS capital_loss
    ,CAST(hours_per_week AS INTEGER) AS hours_per_week
    ,CASE WHEN TRIM(native_country)	='Cambodia' THEN 1 ELSE 0 END AS  Country_Cambodia
    ,CASE WHEN TRIM(native_country)	='Canada' THEN 1 ELSE 0 END AS  Country_Canada
    ,CASE WHEN TRIM(native_country)	='China' THEN 1 ELSE 0 END AS  Country_China
    ,CASE WHEN TRIM(native_country)	='Columbia' THEN 1 ELSE 0 END AS  Country_Columbia
    ,CASE WHEN TRIM(native_country)	='Cuba' THEN 1 ELSE 0 END AS  Country_Cuba
    ,CASE WHEN TRIM(native_country)	='Dominican-Republic' THEN 1 ELSE 0 END AS  Country_Dominican_Republic
    ,CASE WHEN TRIM(native_country)	='Ecuador' THEN 1 ELSE 0 END AS  Country_Ecuador
    ,CASE WHEN TRIM(native_country)	='El-Salvador' THEN 1 ELSE 0 END AS  Country_El_Salvador
    ,CASE WHEN TRIM(native_country)	='England' THEN 1 ELSE 0 END AS  Country_England
    ,CASE WHEN TRIM(native_country)	='France' THEN 1 ELSE 0 END AS  Country_France
    ,CASE WHEN TRIM(native_country)	='Germany' THEN 1 ELSE 0 END AS  Country_Germany
    ,CASE WHEN TRIM(native_country)	='Greece' THEN 1 ELSE 0 END AS  Country_Greece
    ,CASE WHEN TRIM(native_country)	='Guatemala' THEN 1 ELSE 0 END AS  Country_Guatemala
    ,CASE WHEN TRIM(native_country)	='Haiti' THEN 1 ELSE 0 END AS  Country_Haiti
    ,CASE WHEN TRIM(native_country)	='Holand-Netherlands' THEN 1 ELSE 0 END AS  Country_Holand_Netherlands
    ,CASE WHEN TRIM(native_country)	='Honduras' THEN 1 ELSE 0 END AS  Country_Honduras
    ,CASE WHEN TRIM(native_country)	='Hong' THEN 1 ELSE 0 END AS  Country_Hong
    ,CASE WHEN TRIM(native_country)	='Hungary' THEN 1 ELSE 0 END AS  Country_Hungary
    ,CASE WHEN TRIM(native_country)	='India' THEN 1 ELSE 0 END AS  Country_India
    ,CASE WHEN TRIM(native_country)	='Iran' THEN 1 ELSE 0 END AS  Country_Iran
    ,CASE WHEN TRIM(native_country)	='Ireland' THEN 1 ELSE 0 END AS  Country_Ireland
    ,CASE WHEN TRIM(native_country)	='Italy' THEN 1 ELSE 0 END AS  Country_Italy
    ,CASE WHEN TRIM(native_country)	='Jamaica' THEN 1 ELSE 0 END AS  Country_Jamaica
    ,CASE WHEN TRIM(native_country)	='Japan' THEN 1 ELSE 0 END AS  Country_Japan
    ,CASE WHEN TRIM(native_country)	='Laos' THEN 1 ELSE 0 END AS  Country_Laos
    ,CASE WHEN TRIM(native_country)	='Mexico' THEN 1 ELSE 0 END AS  Country_Mexico
    ,CASE WHEN TRIM(native_country)	='Nicaragua' THEN 1 ELSE 0 END AS  Country_Nicaragua
    ,CASE WHEN TRIM(native_country)	='Outlying-US(Guam-USVI-etc)' THEN 1 ELSE 0 END AS  Country_Outlying_US
    ,CASE WHEN TRIM(native_country)	='Peru' THEN 1 ELSE 0 END AS  Country_Peru
    ,CASE WHEN TRIM(native_country)	='Philippines' THEN 1 ELSE 0 END AS  Country_Philippines
    ,CASE WHEN TRIM(native_country)	='Poland' THEN 1 ELSE 0 END AS  Country_Poland
    ,CASE WHEN TRIM(native_country)	='Portugal' THEN 1 ELSE 0 END AS  Country_Portugal
    ,CASE WHEN TRIM(native_country)	='Puerto-Rico' THEN 1 ELSE 0 END AS  Country_Puerto_Rico
    ,CASE WHEN TRIM(native_country)	='Scotland' THEN 1 ELSE 0 END AS  Country_Scotland
    ,CASE WHEN TRIM(native_country)	='South' THEN 1 ELSE 0 END AS  Country_South
    ,CASE WHEN TRIM(native_country)	='Taiwan' THEN 1 ELSE 0 END AS  Country_Taiwan
    ,CASE WHEN TRIM(native_country)	='Thailand' THEN 1 ELSE 0 END AS  Country_Thailand
    ,CASE WHEN TRIM(native_country)	='Trinadad&Tobago' THEN 1 ELSE 0 END AS  Country_TrinadadTobago
    ,CASE WHEN TRIM(native_country)	='United-States' THEN 1 ELSE 0 END AS  Country_United_States
    ,CASE WHEN TRIM(native_country)	='Vietnam' THEN 1 ELSE 0 END AS  Country_Vietnam
    ,CASE WHEN TRIM(native_country)	='Yugoslavia' THEN 1 ELSE 0 END AS  Country_Yugoslavia
    ,CASE WHEN TRIM(income_label)	='>50K' THEN 1 ELSE 0 END AS  target_output
    FROM stg_adult_training
    WHERE TRIM(workclass) <> ''
    AND TRIM(occupation) <> ''
    AND TRIM(native_country) <> ''
    ;
    """
    
    #Execute Query Statement
    orig_train = pgquery (getConnection(), query_stmt, None)
    
    return pd.DataFrame(orig_train)

def getTestDataframe():
    print("Retrieving test data")
    query_stmt ="""
    SELECT 
     cast(age AS INTEGER)
    ,CASE WHEN TRIM(workclass)='Federal-gov' THEN 1 ELSE 0 END AS  Federal_gov
    ,CASE WHEN TRIM(workclass)='Local-gov' THEN 1 ELSE 0 END AS  Local_gov
    --,CASE WHEN TRIM(workclass)='Never-worked' THEN 1 ELSE 0 END AS  Never_worked
    ,CASE WHEN TRIM(workclass)='Private' THEN 1 ELSE 0 END AS  Private
    ,CASE WHEN TRIM(workclass)='Self-emp-inc' THEN 1 ELSE 0 END AS  Self_emp_inc
    ,CASE WHEN TRIM(workclass)='Self-emp-not-inc' THEN 1 ELSE 0 END AS  Self_emp_not_inc
    ,CASE WHEN TRIM(workclass)='State-gov' THEN 1 ELSE 0 END AS  State_gov
    ,CASE WHEN TRIM(workclass)='Without-pay' THEN 1 ELSE 0 END AS  Without_pay
    ,cast(fnlwgt AS INTEGER)
    ,CAST(education_num AS INTEGER)
    ,CASE WHEN TRIM(marital_status) IN ('Married-AF-spouse', 'Married-civ-spouse') THEN 1
          WHEN TRIM(relationship) IN ('Husband', 'Wife') THEN 1
          ELSE 0 END AS  Married
    ,CASE WHEN TRIM(occupation)	='Adm-clerical' THEN 1 ELSE 0 END AS  Adm_clerical
    ,CASE WHEN TRIM(occupation)	='Armed-Forces' THEN 1 ELSE 0 END AS  Armed_Forces
    ,CASE WHEN TRIM(occupation)	='Craft-repair' THEN 1 ELSE 0 END AS  Craft_repair
    ,CASE WHEN TRIM(occupation)	='Exec-managerial' THEN 1 ELSE 0 END AS  Exec_managerial
    ,CASE WHEN TRIM(occupation)	='Farming-fishing' THEN 1 ELSE 0 END AS  Farming_fishing
    ,CASE WHEN TRIM(occupation)	='Handlers-cleaners' THEN 1 ELSE 0 END AS  Handlers_cleaners
    ,CASE WHEN TRIM(occupation)	='Machine-op-inspct' THEN 1 ELSE 0 END AS  Machine_op_inspct
    ,CASE WHEN TRIM(occupation)	='Other-service' THEN 1 ELSE 0 END AS  Other_service
    ,CASE WHEN TRIM(occupation)	='Priv-house-serv' THEN 1 ELSE 0 END AS  Priv_house_serv
    ,CASE WHEN TRIM(occupation)	='Prof-specialty' THEN 1 ELSE 0 END AS  Prof_specialty
    ,CASE WHEN TRIM(occupation)	='Protective-serv' THEN 1 ELSE 0 END AS  Protective_serv
    ,CASE WHEN TRIM(occupation)	='Sales' THEN 1 ELSE 0 END AS  Sales
    ,CASE WHEN TRIM(occupation)	='Tech-support' THEN 1 ELSE 0 END AS  Tech_support
    ,CASE WHEN TRIM(occupation)	='Transport-moving' THEN 1 ELSE 0 END AS  Transport_moving
    ,CASE WHEN TRIM(race)	='Amer-Indian-Eskimo' THEN 1 ELSE 0 END AS  Amer_Indian_Eskimo
    ,CASE WHEN TRIM(race)	='Asian-Pac-Islander' THEN 1 ELSE 0 END AS  Asian_Pac_Islander
    ,CASE WHEN TRIM(race)	='Black' THEN 1 ELSE 0 END AS  Black
    ,CASE WHEN TRIM(race)	='Other' THEN 1 ELSE 0 END AS  Other
    ,CASE WHEN TRIM(race)	='White' THEN 1 ELSE 0 END AS  White
    ,CASE WHEN TRIM(sex)	='Female' THEN 1 ELSE 0 END AS  Gender
    ,CAST(capital_gain AS INTEGER) AS capital_gain
    ,CAST(capital_loss AS INTEGER) AS capital_loss
    ,CAST(hours_per_week AS INTEGER) AS hours_per_week
    ,CASE WHEN TRIM(native_country)	='Cambodia' THEN 1 ELSE 0 END AS  Country_Cambodia
    ,CASE WHEN TRIM(native_country)	='Canada' THEN 1 ELSE 0 END AS  Country_Canada
    ,CASE WHEN TRIM(native_country)	='China' THEN 1 ELSE 0 END AS  Country_China
    ,CASE WHEN TRIM(native_country)	='Columbia' THEN 1 ELSE 0 END AS  Country_Columbia
    ,CASE WHEN TRIM(native_country)	='Cuba' THEN 1 ELSE 0 END AS  Country_Cuba
    ,CASE WHEN TRIM(native_country)	='Dominican-Republic' THEN 1 ELSE 0 END AS  Country_Dominican_Republic
    ,CASE WHEN TRIM(native_country)	='Ecuador' THEN 1 ELSE 0 END AS  Country_Ecuador
    ,CASE WHEN TRIM(native_country)	='El-Salvador' THEN 1 ELSE 0 END AS  Country_El_Salvador
    ,CASE WHEN TRIM(native_country)	='England' THEN 1 ELSE 0 END AS  Country_England
    ,CASE WHEN TRIM(native_country)	='France' THEN 1 ELSE 0 END AS  Country_France
    ,CASE WHEN TRIM(native_country)	='Germany' THEN 1 ELSE 0 END AS  Country_Germany
    ,CASE WHEN TRIM(native_country)	='Greece' THEN 1 ELSE 0 END AS  Country_Greece
    ,CASE WHEN TRIM(native_country)	='Guatemala' THEN 1 ELSE 0 END AS  Country_Guatemala
    ,CASE WHEN TRIM(native_country)	='Haiti' THEN 1 ELSE 0 END AS  Country_Haiti
    ,CASE WHEN TRIM(native_country)	='Holand-Netherlands' THEN 1 ELSE 0 END AS  Country_Holand_Netherlands
    ,CASE WHEN TRIM(native_country)	='Honduras' THEN 1 ELSE 0 END AS  Country_Honduras
    ,CASE WHEN TRIM(native_country)	='Hong' THEN 1 ELSE 0 END AS  Country_Hong
    ,CASE WHEN TRIM(native_country)	='Hungary' THEN 1 ELSE 0 END AS  Country_Hungary
    ,CASE WHEN TRIM(native_country)	='India' THEN 1 ELSE 0 END AS  Country_India
    ,CASE WHEN TRIM(native_country)	='Iran' THEN 1 ELSE 0 END AS  Country_Iran
    ,CASE WHEN TRIM(native_country)	='Ireland' THEN 1 ELSE 0 END AS  Country_Ireland
    ,CASE WHEN TRIM(native_country)	='Italy' THEN 1 ELSE 0 END AS  Country_Italy
    ,CASE WHEN TRIM(native_country)	='Jamaica' THEN 1 ELSE 0 END AS  Country_Jamaica
    ,CASE WHEN TRIM(native_country)	='Japan' THEN 1 ELSE 0 END AS  Country_Japan
    ,CASE WHEN TRIM(native_country)	='Laos' THEN 1 ELSE 0 END AS  Country_Laos
    ,CASE WHEN TRIM(native_country)	='Mexico' THEN 1 ELSE 0 END AS  Country_Mexico
    ,CASE WHEN TRIM(native_country)	='Nicaragua' THEN 1 ELSE 0 END AS  Country_Nicaragua
    ,CASE WHEN TRIM(native_country)	='Outlying-US(Guam-USVI-etc)' THEN 1 ELSE 0 END AS  Country_Outlying_US
    ,CASE WHEN TRIM(native_country)	='Peru' THEN 1 ELSE 0 END AS  Country_Peru
    ,CASE WHEN TRIM(native_country)	='Philippines' THEN 1 ELSE 0 END AS  Country_Philippines
    ,CASE WHEN TRIM(native_country)	='Poland' THEN 1 ELSE 0 END AS  Country_Poland
    ,CASE WHEN TRIM(native_country)	='Portugal' THEN 1 ELSE 0 END AS  Country_Portugal
    ,CASE WHEN TRIM(native_country)	='Puerto-Rico' THEN 1 ELSE 0 END AS  Country_Puerto_Rico
    ,CASE WHEN TRIM(native_country)	='Scotland' THEN 1 ELSE 0 END AS  Country_Scotland
    ,CASE WHEN TRIM(native_country)	='South' THEN 1 ELSE 0 END AS  Country_South
    ,CASE WHEN TRIM(native_country)	='Taiwan' THEN 1 ELSE 0 END AS  Country_Taiwan
    ,CASE WHEN TRIM(native_country)	='Thailand' THEN 1 ELSE 0 END AS  Country_Thailand
    ,CASE WHEN TRIM(native_country)	='Trinadad&Tobago' THEN 1 ELSE 0 END AS  Country_TrinadadTobago
    ,CASE WHEN TRIM(native_country)	='United-States' THEN 1 ELSE 0 END AS  Country_United_States
    ,CASE WHEN TRIM(native_country)	='Vietnam' THEN 1 ELSE 0 END AS  Country_Vietnam
    ,CASE WHEN TRIM(native_country)	='Yugoslavia' THEN 1 ELSE 0 END AS  Country_Yugoslavia
    ,CASE WHEN TRIM(income_label)	='>50K.' THEN 1 ELSE 0 END AS  target_output
    FROM stg_adult_test
    WHERE TRIM(workclass) <> ''
    AND TRIM(occupation) <> ''
    AND TRIM(native_country) <> ''
    """
    
    #Execute Query Statement
    orig_test = pgquery (getConnection(), query_stmt, None)
    
    return pd.DataFrame(orig_test)    

def executeClassifier(clf, X, Y, X_test, Y_test):
    start = datetime.datetime.now()
    clf.fit(X, Y)
    Y_predict = clf.predict(X_test)
    end = datetime.datetime.now()
    elapsed = (end - start).total_seconds()
        
    accuracy = accuracy_score(Y_test, Y_predict)
    mcorr = matthews_corrcoef(Y_test, Y_predict)
    
    print("accuracy:", accuracy)
    print("matthew's correlation coefficient:", mcorr)
    print("f1 scores:", f1_score(Y_test, Y_predict,average=None))
    print("elpased time in seconds:", elapsed)
    
    print("\nconfusion matrix:")
    print(confusion_matrix(Y_test, Y_predict))
    print(classification_report_imbalanced(Y_test, Y_predict))
    return Y_predict, accuracy, mcorr, elapsed

def getData():
    dfTrain = getTrainDataframe()
    dfTest = getTestDataframe()
    
    X_train = dfTrain.iloc[:,:-1]
    Y_train = np.array(dfTrain.iloc[:,-1:]).ravel()
    
    X_test = dfTest.iloc[:,:-1]
    Y_test = np.array(dfTest.iloc[:,-1:]).ravel()    
    
    return X_train, Y_train, X_test, Y_test
    



if __name__ == '__main__':
    usage = "usage: %prog [options] arg1 arg2"
    parser = OptionParser(usage=usage)    
    parser.add_option("-c", "--classifier", dest="classifier", help="name of the classifier. Valid values: AdaBoost, GradientBoost, LogisticRegression, " +
         "XGBClassifier, BaggingClassifier, LinearSVM.  Default value: ALL ", action="store", type="string", default="ALL")
    parser.add_option("-m", '--mingw64', dest="mingw64", help="the path to mingw-w64 bin needed to run xgboost. Default is " +
                      "C:\\Program Files\\mingw-w64\\x86_64-7.1.0-posix-seh-rt_v5-rev0\\mingw64\\bin", action="store", type="string"
                      , default="C:\\Program Files\\mingw-w64\\x86_64-7.1.0-posix-seh-rt_v5-rev0\\mingw64\\bin")
    (options, args) = parser.parse_args()
    clfa = options.classifier

    classifiers = dict()
    classifiers["AdaBoost"] = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=50, learning_rate=1, algorithm='SAMME.R')
    classifiers["GradientBoost"] = GradientBoostingClassifier(max_depth=6, n_estimators=100, learning_rate=0.1, loss='exponential')
    classifiers["LogisticRegression"] = LogisticRegression(penalty='l1', max_iter=100, C=1, class_weight=None)
    classifiers["BaggingClassifier"] = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=14), n_estimators=700, max_features=60, max_samples=1.0)
    classifiers["LinearSVM"] = LinearSVC(dual=False, fit_intercept=False, C=10, class_weight=None)

    if (clfa == "ALL" or clfa == "XGBClassifier"): 
        print("adding to PATH environment variable", options.mingw64)
        if os.path.exists(options.mingw64):      
            os.environ["PATH"] = options.mingw64 + ";" + os.environ["PATH"]
            import xgboost as xgb
       
            classifiers["XGBClassifier"] = xgb.XGBClassifier(max_depth=6, n_estimators=100, learning_rate=0.1, objective='reg:logistic')
        else:
            print(options.mingw64, "does not exists!!")
            raise

    x_train, y_train, x_test, y_test = getData()
    if (clfa == "ALL"):
        sorted_c = sorted(classifiers.keys())
        resultByAccuracy = []
        resultByElapsed = []
        resultByCorr = []
        for name in sorted_c:
            print("\nExecuting", name)
            clf = classifiers[name]
            Y_predict, accuracy, mcorr, elapsed = executeClassifier(clf, x_train, y_train, x_test, y_test)
            resultByAccuracy.append((accuracy, name))
            resultByElapsed.append((elapsed, name))
            resultByCorr.append((mcorr,  name))
        
        bestByAccurcy = max(resultByAccuracy)
        bestByElapsed = min(resultByElapsed)
        bestByCorr = max(resultByCorr)
        
        print("\nHIGHEST ACCURACY:", bestByAccurcy[1], "at", bestByAccurcy[0])
        print("FASTEST IN SECONDS:", bestByElapsed[1], "in", bestByElapsed[0])
        print("HIGHEST MCC:", bestByCorr[1], "at", bestByCorr[0])

    elif (clfa in classifiers):
        clf = classifiers[clfa]
        executeClassifier(clf, x_train, y_train, x_test, y_test)
    else:
        print("Invalid classifier:", clfa)    
        
    
    

    