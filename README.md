
# Students' performance prediction and clustering with LMS log information

This repository contains the code used in the paper *Massive LMS log data analysis for the early prediction of course-agnostic student performance*.

The repository contains not only the code to train the predictive and clustering models, but also the datasets used to create those models.

## Summary

The paper predicts students’ performance in solving LMS assignments by using the log files of a Moodle LMS.  We predict students’ performance at 10%, 25%, 33% and 50% of the course length. Models are created by just using the data available in the LMS before each moment of prediction. 

Our objective is not to predict the exact student’s mark in LMS assignments, but to detect at-risk, fail and excellent students in early stages of the course. That is why we create different classification models for each of those three student groups. We take 2.5 as the threshold mark for at-risk students, and 8.5 for excellent ones. We also take 5.0 as another mark threshold to differentiate pass students from fail ones.

## Algorithms

We use different supervised machine learning algorithms to predict students’ performance and an unsupervised algorithm for student clustering.


### Supervised models

The algorithms used in this experiment where scikit-learn’s:

* `NaiveBayes`
* `DecisionTree`
* `LogisticRegression`
* `SVC`
* `Multi-Layer Perceptron`

With each algorithm, 12 predictive models, considering 4 moments of prediction (10%, 25%, 33% and 50%) and 3 mark thresholds (2.5, 5.0 and 8.5).

We select the best hyperparameters for each algorithm with exhaustive parallel search across common parameter values (`RandomizedSearchCV`), using stratified randomized 3-fold cross validation (`StratifiedKFold`). Accuracy is the metric used to measure the performance of each hyperparameter combination.

These are the hyperparameters chosen for the different algorithms (for those hyperparameters not shown, we chose the default values provided by scikit-learn):

**DecisionTree**
* Mark threshold 2.5:
	- 10%, 25%, 33% and 50%: `splitter = random, presort = False, max_features = None, max_depth = 10, criterion = gini, class_weight = None`.  
* Mark threshold 5.0: 
	- 10%: `splitter = best, presort = False, max_features = sqrt, max_depth = 10, criterion = gini, class_weight = None`.
    - 25% and 50%: `splitter = random, presort = True, max_features = None, max depth = 10, criterion = gini, class_weight = balanced`.
    - 33%: `splitter = random, presort = True, max_features = None, max- depth = 10, criterion = entropy, class_weight = balanced`.     
* Mark threshold 8.5:  
	 - 10%, 25%, 33% and 50%: `splitter = best, presort = True, max_features = None, max depth = 5, criterion = entropy, class_weight = None`.


**NaiveBayes**
* Mark threshold 2.5, 5.0 and 8.5:
    - 10%, 25%, 33% and 50%: `var_smoothing = 1e-09`.


**LogisticRegression**
* Mark thresholds 2.5:  
	 - 10% and 25%: `tol = 0.0001, solver = liblinear, penalty = l1, max_iter = 200`.  
	 - 33%: `tol = 0.001, solver = liblinear, penalty = l2, max_iter = 50`. 
	 - 50%: `tol = 0.01, solver = liblinear, penalty = l2, max_iter= 100`.
* Mark threshold 5.0:
	 - 10%: `tol = 1e-05, solver = liblinear, penalty = l2, max_iter = 200`.
	 - 25% and 33%: `tol = 0.001, solver = liblinear, penalty = l1, max_iter= 100`.
	 - 50%: `tol = 1e-05, solver = liblinear, penalty = l1, max_iter = 200`.
* Mark threshold 8.5:  
	 - 10%: `tol = 1e-05, solver = liblinear, penalty = l2, max_iter = 50`. 
	 - 25%: `tol = 0.01, solver = liblinear, penalty = l1, max_iter = 100`. 
	 - 33%: `tol = 0.0001, solver = liblinear, penalty = l1, max_iter = 100`. 
	 - 50%: `tol = 0.01, solver = liblinear, penalty = l2, max_iter = 100`.


**Multi-Layer Perceptron**
* Mark threshold 2.5:
	 - 10%, 25%, 33% and 50%: `solver = lbfgs, learning_rate = constant, hidden_layer_sizes = 20, alpha = 0.1, activation = tanh`.
* Mark threshold 5.0:
	 - 10% and 50%: `solver = lbfgs, learning_rate = adaptive, hidden_layer_sizes = 20, alpha = 0.01, activation = relu`.
	 - 25%: `solver = adam, learning_rate = constant, hidden_layer_sizes = 20, alpha = 0.001, activation = relu`.
	 - 33%: `solver = lbfgs, learning_rate = invscaling, hidden_layer_sizes = 20, alpha = 0.1, activation = relu`. 
 * Mark threshold 8.5:
	 - 10%: `solver = adam, learning_rate = adaptive, hidden_layer_sizes = 20, alpha = 0.001, activation = relu`.
	 - 25%: `solver = lbfgs, learning_rate = adaptive, hidden_layer_sizes = 20, alpha = 1, activation = relu`.
	 - 33% and 50%: `solver = lbfgs, learning_rate = invscaling, hidden_layer_sizes = 20, alpha = 1, activation = relu`.


**Support Vector Classifier**

* Mark thresholds 2.5, 5.0 and 8.5.
	- 10%, 25%, 33% and 50%: `tol = 0.01, probability = True, kernel = rbf, gamma = scale, cache_size = 4096, C = 1'`.


We store the following metrics to evaluate each model:

* Accuracy.
* AUC.
* Confusion matrix.
* Precision, recall and F1-score (sklearn `classification_report).
* Values to draw ROC curve (True Positive Rate and False Positive Rate points).

### Unsupervised models

For the unsupervised models, we aggregate features of in the dataset with the `FeatureAggregation` class in scikit-learn. The number of features was reduced from 60 to 4.

Student clusters were generated with the K-means algorithm. The optimum number of clusters was chosen using the GAP distance, getting 6 clusters for all the moments of prediction.

## Code

NOTE: it's possible that code needs to be adapted to fit your system path / installation. 

### Supervised training

The code for launching the supervised training is stored in `code/supervised` folder and it receives 3 parameters in this order:

* **Time**: options available are *0.1*, *0.25*, *0.33*, *0.5*.
* **Grade**: options available are *2.5*, *5.0*, *8.5*.
* **Model type**: options available are *nb* (NaiveBayes), *dt* (Decision Tree), *lr* (Logistic Regression), *svc* (SVC), *nn* (Multi-Layer Perceptron).

Dataset used for training these models are stored in the folder [Supervised dataset](#supervised-dataset)

The code for computing the dependent variable (students' performance in solving LMS assignments)
is placed in the `estimate.py` file in the `code/supervised` folder.


### Unsupervised training

The code for launching the supervised training is stored in the `code/unsupervised` folder and it receives 2 parameters in this order:

* **Time**: options available are *0.1*, *0.25*, *0.33*, *0.5*.
* **Data path**: path were input data is stored.

Dataset used for training these models are stored in the folder  [Unsupervised dataset](#unsupervised-dataset).


## Data 

This dataset is generated from Moodle log information. The original file is too big to be uploaded here, but it could be downloaded from [dump_mysql](https://storage.googleapis.com/dissertation-data/dissertation-export/mysql-export)

This file above is a MySQL dump that has all the necessary information to generate the datasets shown in the following folders.

If you prefer a PostgreSQL dump, you could download it from [dump_postgresql](https://storage.googleapis.com/dissertation-data/dissertation-export/192.168.25.168_moodle2014anonimo_2015-10-26_18h28m00s.pg_dump.sql.zip).

### Supervised dataset

There are several datasets generated for each time moment of prediction and mark threshold. They are stored in the folder `data/supervised`. They are binary files serialized with Python’s `pickle` package. The naming convention is:

* `clean_df_TIME_GRADE.pkl` where TIME is the first parameter of the process and GRADE is the second parameter of the process.

### Unsupervised dataset

There are several datasets generated for each moment of prediction. They are stored in the folder `data/unsupervised`. They are stored in *csv* format. The naming convention is:

* `input_TIME.csv` where TIME is the first parameter of the process.
