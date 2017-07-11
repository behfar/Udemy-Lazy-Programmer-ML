import numpy as np
import matplotlib.pyplot as plt

def make_poly(X, deg):
    N = len(X)
    data = [np.ones(N)]
    for d in range(deg):
        data.append(X**(d+1))
    return np.stack(data).T

def fit(X, Y):
    w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
    return w

def fit_and_display(X, Y, sample, deg):
    N = len(X)
    train_idx = np.random.choice(N, int(N*sample))
    X_train = X[train_idx]
    Y_train = Y[train_idx]

    # fit polynomial regression
    X_train_poly = make_poly(X_train, deg)
    w = fit(X_train_poly, Y_train)

    # display the polynomial regression
    X_poly = make_poly(X, deg)
    Y_hat = X_poly.dot(w)
    plt.plot(X, Y)
    plt.plot(X, Y_hat)
    plt.scatter(X_train, Y_train)
    plt.title("deg = %d" % deg)
    plt.show()

def get_mse(Y, Y_hat):
    d = Y - Y_hat
    return d.dot(d) / len(d)

def plot_train_vs_test_curves(X, Y, sample=0.2, max_deg=20):
    N = len(X)
    train_idx = np.random.choice(N, int(N*sample))
    X_train = X[train_idx]
    Y_train = Y[train_idx]

    test_idx = list(set(range(N)) - set(train_idx))
    X_test = X[test_idx]
    Y_test = Y[test_idx]

    mse_trains = []
    mse_tests = []
    degs = range(1, max_deg+1)
    for deg in degs:
        X_train_poly = make_poly(X_train, deg)
        w = fit(X_train_poly, Y_train)
        Y_hat_train = X_train_poly.dot(w)

        X_test_poly = make_poly(X_test, deg)
        Y_hat_test = X_test_poly.dot(w)

        mse_trains.append(get_mse(Y_train, Y_hat_train))
        mse_tests.append(get_mse(Y_test, Y_hat_test))
    
    # look at train vs test curves
    plt.plot(degs, mse_trains, label="train mse")
    plt.plot(degs, mse_tests, label="test mse")
    plt.legend()
    plt.show()

    # look at train curve by itself
    plt.plot(degs, mse_trains, label="train mse")
    plt.legend()
    plt.show()

# make up some data looking like a sine wave
N = 100
X = np.linspace(0, 6*np.pi, N)
Y = np.sin(X)

# fit regresion and display for different degree polynomials
sample = 0.1 # using 10% of data for training
for deg in range(5,10):
    fit_and_display(X, Y, sample, deg)

# look at the train vs test learning curves
plot_train_vs_test_curves(X, Y, sample=sample, max_deg=20)
