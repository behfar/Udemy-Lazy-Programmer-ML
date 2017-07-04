import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# load the data
X = []
Y = []
for line in open("data_2d.csv"):
    x1, x2, y = line.split(',')
    X.append([float(x1), float(x2), 1])
    Y.append(float(y))

# turm X and Y into numpy arrays
X = np.array(X)
Y = np.array(Y)

# plot the data to see what it looks like
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)
plt.show()

# calcuate regression weights
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Y_hat = np.dot(X, w)

# compute R-squared
d1 = Y - Y_hat
SS_res = d1.dot(d1)
d2 = Y - Y.mean()
SS_tot = d2.dot(d2)
R2 = 1 - (SS_res / SS_tot)
print("R-squared is ", R2)

# plot the data and regression plane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)
X_mesh, Y_mesh = np.meshgrid(X.sort(), Y.sort())
Z_mesh = X_mesh # change this to the real Z
ax.plot_surface(X_mesh, Y_mesh, Z_mesh)
plt.show()
