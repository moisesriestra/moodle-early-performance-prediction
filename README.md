# Moodle early performance prediction

This repository contains the code which was generated to perform the different experiments explained in the submitted paper *Massive LMS log data analysis for the early prediction of course-agnostic student performance* [1].

The repository contains not also the code to train models (prediction and clustering) but also the datasets used to train these algorithms to make the experiment reproductible.

## Summary

The paper has 2 different approaches based on the use of actions of students inside an e-learning platform to try to:

1. Predict if the student will pass or not the course (or certain grade) based on the actions he/she performs in the first stages of the course.
2. Analyze if there is any relationship between the actions of a student in the course and the final grade.

Both analysis were performed in different moments during the course taking into account that different courses could have different durations, so we take 4 different moments to perform the analysis which were 10%, 25%, 33% and 50% of the course.

## Algorithms

This sections contains the list of algortihms used in both techniques, supervised and unsupervised, and how they were trained and with which dataset.

### Supervised models

The algortihms tested in this experiment where:

* NaiveBayes
* DecisionTree
* LogisticRegression
* SVC
* Multi Layer Percenptron

Each of them was trained in the moments defined previously and with 4 different cut-offs of grades (2.5, 5.0 and 8.5) so the total amount of models trained were 60 models. For these training processes, the following hyper-parameters were included inside a GridSearch with cross-validation to find the best fit of the model.

**NaiveBayes**
```python
params = {
	'var_smoothing': [1e-09, 1e-08, 1e-010]
}
```

**DecisionTree**
```python
params = {
	'criterion': ['gini', 'entropy'],
	'splitter': ['best', 'random'],
	'max_depth': [None, 5, 10, 15],
	'max_features': [None, 'auto', 'sqrt', 'log2'],
	'class_weight': [None, 'balanced'],
	'presort': [True, False]
}
```

**LogisticRegression**
```python
params = {
	'penalty': ['l2', 'l1'],
	'tol': [1e-2, 1e-3, 1e-4, 1e-5],
	'solver': ['liblinear'],
	'max_iter': [100, 50, 200]
}
```

**SVC**
```python
params = {
	'C': [1],
	'kernel': ['rbf'],
	'gamma': ['scale'],
	'tol': [1e-2],
	'probability': [True],
	'cache_size': [1024 * 4]
}
```

**MultiLayer Perceptron**
```python
params = {
	'hidden_layer_sizes': [20, (20, 20)],
	'activation': ['identity', 'relu', 'tanh', 'relu'],
	'solver': ['adam', 'sgd', 'lbfgs'],
	'alpha': [1, 0.1, 0.01, 0.001],
	'learning_rate': ['constant', 'invscaling', 'adaptive']
}
```

Some different metrics were shown in the log of each model but best model selection was based on the accuracy of the models. The complete list of metrics / values stored during the training is:

* Accuracy
* AUC
* Confusion matrix
* Precison, recall and F1-score (sklearn classification_report)
* Values to draw ROC curve (True Positive Rate and False Positive Rate points)

### Unsupervised models

The process developed for the unsupervised models aggregate the variables stored generated to obtain a subset of variables that should be easily explained when analizing the clusters. This aggregation was performed using the FeattreAggregation package in scikit-learn library and the resulting number of features were 4 when the initial dataset has up to 60 variables.

Finally, generate the clusters using the KMeans algorithm. The number of clusters was selected between 1 to 10 and the optimum number of clusters using the GAP distance and fianlly we obtained that the best number of clusters for all the experiments is 6.

## Code

NOTE: it's possible that code needs to be adapted to fit your system path / installation. Some paths should be adapted.

### Supervised training

The code for launching the supervised training is stored in `code/supervised` folder and it receives 3 parameters in this order:

* **Time**: options available are *0.1*, *0.25*, *0.33*, *0.5*
* **Grade**: options available are *2.5*, *5.0*, *8.5*
* **Model type**: options available are *nb* (NaiveBayes), *dt* (Decision Tree), *lr* (Logistic Regression), *svc* (SVC), *nn* (Multi Layer Perceptron)

Dataset used for training these models are stored in folder explained [Supervised dataset](#supervised-dataset)

### Unsupervised training

The code for launching the supervised training is stored in `code/unsupervised` folder and it receives 2 parameters in this order:

* **Time**: options available are *0.1*, *0.25*, *0.33*, *0.5*
* **Data path**: path were input data is stored.

Dataset used for training these models are stored in folder explained [Unsupervised dataset](#unsupervised-dataset)

## Data 

This dataset is based on processing several log information stored in Moodle database. The original file is to big to be uploaded here but it could be downloaded from the following link: [dump_mysql](https://storage.googleapis.com/dissertation-data/dissertation-export/mysql-export)

This file is a MySQL dump that has all the necessary information to generate the datasets shown in the following folders.

If you prefer a PostgreSQL dump it could be downloaded from [dump_postgresql](https://storage.googleapis.com/dissertation-data/dissertation-export/192.168.25.168_moodle2014anonimo_2015-10-26_18h28m00s.pg_dump.sql.zip)

### Supervised dataset

There are several datasets generated for each time moment and each grade. They are stored into the folder `data/supervised`.

They are stored in **Pickle** so it can be easyly read using `pandas` library. The convention for naming is:

* `clean_df_TIME_GRADE.pkl` where TIME is the first parameter of the process and GRADE is the second parameter of the process

### Unsupervised dataset

There are several datasets generated for each time moment. They are stored into the folder `data/unsupervised`.

They are stored in **CSV** so it can be easyly read using `pandas` library. The convention for naming is:

* `input_TIME.csv` where TIME is the first parameter of the process