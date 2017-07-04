import numpy as np
import matplotlib.pyplot as plt

# load the data into X and Y vectors
X = []
Y = []
for line in open('data_1d.csv'):
    x, y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

# turn X and Y into numpy arrays
X = np.array(X)
Y = np.array(Y)

# plot to see what the data looks like
plt.scatter(X, Y)
plt.show()

# apply regression solution to fit line
denominator = X.dot(X) - X.mean()*X.sum()
a = ( X.dot(Y) - Y.mean()*X.sum() ) / denominator
b = ( Y.mean()*X.dot(X) - X.mean()*X.dot(Y)) / denominator

# plot data and regression line
Y_hat = a*X + b
plt.scatter(X, Y)
plt.plot(X, Y_hat)
plt.show()

# compute R-squared which is 1 minus (residual sum of squares / total sum of squares)
d1 = Y - Y_hat
SS_res = d1.dot(d1)
d2 = Y - Y.mean()
SS_tot = d2.dot(d2)
R2 = 1 - (SS_res / SS_tot)
print("R2 is ", R2)
