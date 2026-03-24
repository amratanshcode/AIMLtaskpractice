'''Polynomial Regression is also a supervised learning algorithm,
 but instead of a straight line it fits a curved relationship.
 Rel between variable is non-linear '''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Dataset
X = np.array([1,2,3,4,5]).reshape(-1,1)
y = np.array([1,4,9,16,25])

# Polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Model
model = LinearRegression()
model.fit(X_poly, y)

# Prediction
y_pred = model.predict(X_poly)

# Plot
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')

plt.title("Polynomial Regression")
plt.xlabel("X")
plt.ylabel("Y")

plt.show()