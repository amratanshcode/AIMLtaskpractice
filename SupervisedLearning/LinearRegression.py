'''Linear Regression is a supervised learning algorithm used to predict a continuous value by fitting a straight line through the data.'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample dataset
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Prediction
y_pred = model.predict(X)

# Plot
plt.scatter(X, y, color="blue")
plt.plot(X, y_pred, color="red")

plt.title("Simple Linear Regression")
plt.xlabel("X")
plt.ylabel("Y")

plt.show()

# Print equation
print("Slope:", model.coef_)
print("Intercept:", model.intercept_)