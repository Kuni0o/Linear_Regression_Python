from regressions import linear_regression as lr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read data from CSV file
data_file = pd.read_csv('linear_regression_test_data.csv')

# Train linear regression model on the data
m, b = lr.train_linear_regression(np.array(data_file['x']), np.array(data_file['y']), 1000, 0.5, 2.7, 2.2)

# Generate linear line using trained coefficients
x_linear = np.array([0, 1])
y_linear = lr.predict(m, b, x_linear)

# Plot the linear line and data points
plt.plot(x_linear, y_linear, color="red")
plt.scatter(data_file['x'], data_file['y'])
plt.show()
