# https://deeplearningcourses.com/c/data-science-linear-regression-in-python
# need to sudo pip install xlrd to use pd.read_excel
# data is from:
# http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/mlr02.html

# The data (X1, X2, X3) are for each patient.
# X1 = systolic blood pressure
# X2 = age in years
# X3 = weight in pounds

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_excel('mlr02.xls')
X = df.as_matrix()

# look at data to see if linear in each input
plt.scatter(X[:,1], X[:,0])
plt.show()
plt.scatter(X[:,2], X[:,0])
plt.show()

# compute regression weights
df['ones'] = 1
Y = df['X1']
X = df[['ones', 'X2', 'X3']]
X2_only = df[['ones', 'X2']]
X3_only = df[['ones', 'X3']]
df['rand'] = np.random.randn(df['ones'].size)
X_with_rand = df[['ones','X2','X3','rand']]

def get_R2(X, Y):
    w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
    Y_hat = X.dot(w)
    
    d1 = Y - Y_hat
    d2 = Y - Y.mean()
    R2 = 1 - (d1.dot(d1) / d2.dot(d2))
    return R2

print("R2 for X2 only:", get_R2(X2_only, Y))
print("R2 for X3 only:", get_R2(X3_only, Y))
print("R2 for X2 and X3:", get_R2(X, Y))
print("R2 for X_with_rand:", get_R2(X_with_rand, Y))