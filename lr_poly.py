import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []

# load the data
for line in open('data_poly.csv'):
    x, y = line.split(',')
    x = float(x)
    X.append([1, x, x*x])
    Y.append(float(y))

# convert to numpy arrays
X = np.array(X)
Y = np.array(Y)

# let's plot to see what it looks like
plt.scatter(X[:,1], Y)
plt.show()

# compute regression weights
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Y_hat = np.dot(X, w)

# plot regression
plt.scatter(X[:,1], Y, color='blue')
plt.plot(X[:,1], Y_hat, color='red')
plt.show()
