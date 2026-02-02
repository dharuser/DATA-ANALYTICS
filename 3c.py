from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([1, 2, 3, 4]).reshape(-1, 1)  # Independent variable
y = np.array([300, 500, 700, 900])         # Dependent variable

model = LinearRegression()
model.fit(X, y)

print("Predicted sales:", model.predict([[5]]))

