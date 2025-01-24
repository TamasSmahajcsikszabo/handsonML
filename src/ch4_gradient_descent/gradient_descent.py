'''
    Gradient Descent (GD):
        - tweaking parameters iteratively to minimize a cost function
        - measures the local gradient if the error function with regard to a
        parameter vectorm THETA, and it goes in the direction of the
        descending gradient
        - stops at a minimum

        1. starts with random initialization (assigns random values to THETA)
        2. updates THETA for minimizing the cost function
        3. the steps it takes are defined by the learning rate
        4. converge depends on the form of the cost function
        5. local and global minimum
        6. early stopping
        7. the shape of the cost function also depends on the scale of the
        features: it can be elongated, which impacts convergence time and direction

        Batch GD:
        --------
         - the gradient of a loss function with regards to each model
         parameter:
                how much the function woll change if any Theta_j changes a certain
                amount (partial derivatie)

                for a single Theta_j parameter:
                Deriv / (Deriv * Theta_j) MSE(THETA) = 2/m * Sum(THETA^Tx(i) - y(i)) x_j(i)

                for all parameters a gradient vector can be computed
                nabla (del)

                Del_theta MSE  (THETA)
'''

# Batch Gradient Descent
# takes the whole X dataset at every step
import numpy as np
from numpy.random import random

eta = 0.1
n_iterations = 1000
m = 100
X = 2 * np.random.rand(m,1)
Y = 4 + 3 * X + np.random.randn(m, 1)
X_b = np.c_[np.ones((m,1)),X]

theta = np.random.randn(2, 1)

for i in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - Y)
    theta = theta - eta * gradients

# Stochastic Gradient Descent
# selects random instance
# longer taining, the cost function decreases only on average
# bounces at minimum, around it
# this helps avoiding local minima
# solution: reduce learning rate = simulated annealing
# learning schedule = the function responsible for the learning rate

n_epochs = 50
t0, t1 = 5, 50

def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2,1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = Y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients

print(f"Theta: {theta}")

from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=50, penalty=None, eta0=0.1)
sgd_reg.fit(X, Y.ravel())
sgd_reg.intercept_
sgd_reg.coef_

# Minibatch Gradient Descent

