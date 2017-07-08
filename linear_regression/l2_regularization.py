import numpy as np
import matplotlib.pyplot as plt

# number of samples
N = 50

X = np.linspace(0,10,N)
Y = 0.5*X + np.random.randn(N)

# make the last two Y's outliers
Y[-1] += 30
Y[-2] += 30

plt.scatter(X, Y)
plt.show()

# add bias column to X and calculate w using maximum likelihood solution
X = np.vstack([np.ones(N), X]).T
w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
Y_hat_ml = X.dot(w_ml)
plt.scatter(X[:,1],Y)
plt.plot(X[:,1],Y_hat_ml)
plt.show()

# do the same with L2 regularization and calculate w using maximum a posteriori solution
l2 = 1000.0
w_map = np.linalg.solve(l2*np.eye(2) + X.T.dot(X), X.T.dot(Y))
Y_hat_map = X.dot(w_map)
plt.scatter(X[:,1], Y)
plt.plot(X[:,1], Y_hat_ml, label='maximum likelihood')
plt.plot(X[:,1], Y_hat_map, label='map')
plt.legend()
plt.show()
