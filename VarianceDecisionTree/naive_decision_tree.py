from Core.PRef import PRef
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Optional: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a DecisionTreeRegressor with a specified max depth
max_depth = 5  # Adjust the depth as needed
regressor = DecisionTreeRegressor(max_depth=max_depth, random_state=42)

# Train the model
regressor.fit(X_train, y_train)

# Predict and evaluate (optional)
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")


def create_naive_decision_tree(pRef: PRef, maximum_depth: int):
    pass