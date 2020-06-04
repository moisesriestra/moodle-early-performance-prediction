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

Each of them was trained in the moments defined previously and with 4 different cut-offs of grades (2.5, 5.0 and 8.5) so the total amount of models trained were 60 models. For these training processes, the following hyper-parameters were included inside a GridSearch to find the best fit of the model.

| Model                 | List of hyperparameters |
|-----------------------|-------------------------|
| NaiveBayes            | <pre lang="python">{'var_smoothing': [1e-09, 1e-08, 1e-010]} </pre>|
| DecisionTree          | <pre lang="python">{<br> 'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'],<br> 'max_depth': [None, 5, 10, 15],<br> 'max_features': [None, 'auto', 'sqrt', 'log2'],<br> 'class_weight': [None, 'balanced'], <br> 'presort': [True, False]<br>}</pre> |
| LogisticRegression    | <pre lang="python">{<br> 'penalty': ['l2', 'l1'],<br> 'tol': [1e-2, 1e-3, 1e-4, 1e-5],<br> 'solver': ['liblinear'],<br> 'max_iter': [100, 50, 200]<br>}</pre> |
| SVC                   | <pre lang="python">{<br> 'C': [1],<br> 'kernel': ['rbf'],<br> 'gamma': ['scale'],<br> 'tol': [1e-2],<br> 'probability': [True],<br> 'cache_size': [1024 * 4]<br>}</pre> |
| MultiLayer Perceptron | <pre lang="python">{<br> 'hidden_layer_sizes': [20, (20, 20)],<br> 'activation': ['identity', 'relu', 'tanh', 'relu'],<br> 'solver': ['adam', 'sgd', 'lbfgs'],<br> 'alpha': [1, 0.1, 0.01, 0.001],<br> 'learning_rate': ['constant', 'invscaling', 'adaptive']<br>}</pre> |

### Unsupervised models

## Code

## Data 