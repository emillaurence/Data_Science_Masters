Prerequisites:
  xgboost
  mingw-w64

Installation on xgboost and mingw-w64 with Anaconda: https://www.ibm.com/developerworks/community/blogs/jfp/entry/Installing_XGBoost_For_Anaconda_on_Windows?lang=en


Running the code:

There are two arguments passed:
  -c the classifier to run.  Can be any of the following values, default is ALL:
      	AdaBoost		
      	GradientBoost		
	LogisticRegression	
	XGBClassifier
	BaggingClassifier
	LinearSVM


  -m  the mingw64 installation bin directory.  This is needed to run the C++ library,
	libxgboost.dll, that comes with xgboost installation.  Importing xgboost
	directly results in an error even if the path was added to system PATH variable.  
	Thus, this is added to PATH environment at run time.  

        Note the default path is "C:\Program Files\mingw-w64\x86_64-7.1.0-posix-seh-rt_v5-rev0\mingw64\bin"

	Note that this is only needed for "XGBClassifier" and "ALL"



To run a classifier, i.e. AdaBoost:
   python.exe comp5318_assign2.py -c AdaBoost

To run ALL classifiers:
   python.exe comp5318_assign2.py -c ALL -m <mingw64 bin>