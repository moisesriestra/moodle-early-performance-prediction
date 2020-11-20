#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file implements the different estimates of the target (dependent) variable explained in the article.
"""

import pandas as pd
from typing import List
from scipy.stats import pearsonr, spearmanr, kendalltau
import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

#  Loads the final marks merged with those taken from the LMS
marks = pd.read_csv("merge.csv")


# Normalize final marks (sies) between 0 and 1
marks['sies'] = marks['sies'] / 10.0

#print(marks.columns)
print(marks)
print(marks.describe(include="all"))


print("\nIs there any NaN in Marks? {}.".format(marks.isnull().values.any()))


def correlation(x: List[float], y: List[float], corr_method: str):
    if corr_method.lower() == "pearson":
        return pearsonr(x, y)
    if corr_method.lower() == "spearman":
        return spearmanr(x, y)
    if corr_method.lower() == "kendalltau":
        return kendalltau(x, y)
    raise ValueError("Unknown method")


def adjusted_r2_score(true_values: List[float], pred_values: List[float], n_rows: int, n_features: int):
    r2_value = r2_score(true_values, pred_values)
    return 1 - (1 - r2_value) * (n_rows - 1) / (n_rows - n_features - 1)


def div_zero(num, den):
    if num == 0 or den == 0:
        return 0
    return num / den


def equation1(alpha, mand_sum, mand_count, mand_sub, op_sum, op_count, op_sub: float) -> float:
    return alpha * div_zero(mand_sum, mand_sub) + (1 - alpha) * div_zero(op_sum, op_sub)



def equation2(alpha, mand_sum, mand_count, mand_sub, op_sum, op_count, op_sub: float) -> float:
    return alpha * div_zero(mand_sum, mand_count) + (1 - alpha) * div_zero(op_sum, op_count)


#  Computation of equations for a given alpha
alpha = 0.5
#  print("Eq1\t\t\tEq2\t\t\tEq3\t\t\tEq4\t\t\tSIES")
for index, row in marks.iterrows():
    mand_sum, mand_count, mand_sub, op_sum, op_count, op_sub, sies = row['mand_sum'], row['mand_count'], \
                                row['mand_sub'], row['op_sum'], row['op_count'], row['op_sub'], row['sies']
    eq1 = equation1(alpha, mand_sum, mand_count, mand_sub, op_sum, op_count, op_sub)
    eq2 = equation2(alpha, mand_sum, mand_count, mand_sub, op_sum, op_count, op_sub)
    eq3 = div_zero(mand_sum + op_sum, mand_sub + op_sub)
    eq4 = div_zero(mand_sum + op_sum, mand_count + op_count)
    marks.at[index, 'eq1'], marks.at[index, 'eq2'], marks.at[index, 'eq3'], marks.at[index, 'eq4'] = \
        eq1, eq2, eq3, eq4
    #  print("{:0.6f}\t{:0.6f}\t{:0.6f}\t{:0.6f}\t{:0.6f}".format(equation1, equation2, equation3, equation4, sies))

print(marks)


# Computation of all the correlations
equations = ['eq1', 'eq2', 'eq3', 'eq4']
corr_methods = ['pearson', 'spearman', 'kendalltau']

print(f"Correlations for alpha = {alpha}.")
for corr_method in corr_methods:
    print(f"\tCorrelations ({corr_method}):")
    for equation in equations:
        result = correlation(marks[equation].to_numpy(), marks['sies'].to_numpy(), corr_method)[0]
        print(f"\t\tCorrelation of {equation} and sies: {result}.")
print()


#  Other measures
print(f"Other measures:")
for equation in equations:
    print(f"\tEquation {equation}:")
    mse = mean_squared_error(marks['sies'].to_numpy(), marks[equation].to_numpy())
    print(f"\t\tMSE of {equation}: {mse}.")
    print(f"\t\tRMSE of {equation}: {math.sqrt(mse)}.")
    mae = mean_absolute_error(marks['sies'].to_numpy(), marks[equation].to_numpy())
    print(f"\t\tMAE of {equation}: {mae}.")
    r2_value = r2_score(marks['sies'].to_numpy(), marks[equation].to_numpy())
    print(f"\t\tR2 of {equation}: {r2_value}.")
    adj_r2_score = adjusted_r2_score(marks['sies'].to_numpy(), marks[equation].to_numpy(), marks.shape[0], 1)
    print(f"\t\tAdjusted R2 of {equation}: {adj_r2_score}.")
print()


#  Computation of the alpha that provides the best correlation
equations = ['eq1', 'eq2', 'eq3', 'eq4']
corr_method = "pearson"
max_corr = -1
step = 0.1
for alpha in np.arange(0, 1 + step, step):
    for index, row in marks.iterrows():
        mand_sum, mand_count, mand_sub, op_sum, op_count, op_sub, sies = row['mand_sum'], row['mand_count'], \
                                    row['mand_sub'], row['op_sum'], row['op_count'], row['op_sub'], row['sies']
        eq1 = equation1(alpha, mand_sum, mand_count, mand_sub, op_sum, op_count, op_sub)
        eq2 = equation2(alpha, mand_sum, mand_count, mand_sub, op_sum, op_count, op_sub)
        eq3 = div_zero(mand_sum + op_sum, mand_sub + op_sub)
        eq4 = div_zero(mand_sum + op_sum, mand_count + op_count)
        marks.at[index, 'eq1'], marks.at[index, 'eq2'], marks.at[index, 'eq3'], marks.at[index, 'eq4'] = \
            eq1, eq2, eq3, eq4
    for equation in equations:
        corr = correlation(marks[equation].to_numpy(), marks['sies'].to_numpy(), corr_method)[0]
        #  print(f"alpha {alpha}, eq {equation}, corr {corr}")
        if corr > max_corr:
            max_corr, max_alpha, max_eq = corr, alpha, corr
            #  print(f"-> max alpha: {alpha}, eq: {equation}, corr: {corr}")

print(f"Max correlation: {max_corr} for alpha {max_alpha} and equation {max_eq}.\n")


#  Regression models are also computed to compare their accuracy with the proposed equations

X = marks[['mand_sum', 'mand_count', 'mand_sub', 'op_sum', 'op_count', 'op_sub']]
y = marks['sies']

linear_model = LinearRegression().fit(X, y)
print(f"Accuracy of linear regression model: {linear_model.score(X, y)}.")

elastic_model = ElasticNet().fit(X, y)
print(f"Accuracy of elastic net regression model: {elastic_model.score(X, y)}.")

ridge_model = Ridge(alpha=0.5).fit(X, y)
print(f"Accuracy of ridge regression model: {ridge_model.score(X, y)}.")

lasso_model = Lasso(alpha=0.1).fit(X, y)
print(f"Accuracy of lasso regression model: {lasso_model.score(X, y)}.")

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X = poly.fit_transform(X)
poly_model = LinearRegression().fit(X, y)
print(f"Accuracy of polynomial regression model: {poly_model.score(X, y)}.")


for index, row in marks.iterrows():
    x_values = np.array([[row['mand_sum'], row['mand_count'], row['mand_sub'], row['op_sum'], row['op_count'], row['op_sub']]])
    marks.at[index, 'regression'] = linear_model.predict(x_values)
corr = correlation(marks['regression'].to_numpy(), marks['sies'].to_numpy(), corr_method)[0]
print(f"Correlation of the linear model: {corr}")


mse = mean_squared_error(marks['sies'].to_numpy(), marks['regression'].to_numpy())
print("MSE of linear regression: {}.".format(mse))
print("RMSE of linear regression: {}.".format(math.sqrt(mse)))
mae = mean_absolute_error(marks['sies'].to_numpy(), marks['regression'].to_numpy())
print("MAE of linear regression: {}.".format(mae))
r2_value = r2_score(marks['sies'].to_numpy(), marks['regression'].to_numpy())
print("R2 of linear regression: {}.".format(r2_value))
adj_r2_score = adjusted_r2_score(marks['sies'].to_numpy(), marks['regression'].to_numpy(), marks.shape[0], 1)
print("Adjusted R2 of linear regression: {}.".format(adj_r2_score))

print(f"Coefficients: {linear_model.coef_}.")