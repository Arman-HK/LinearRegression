import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("honeyproduction.csv")
prod_year = df.groupby("year").totalprod.mean().reset_index()
X = prod_year["year"]
X = X.values.reshape(-1, 1)
y = prod_year["totalprod"]
plt.scatter(X, y)
reg = linear_model.LinearRegression()
reg.fit(X, y)
y_predict = reg.predict(X)
plt.plot(X, y_predict)
X_future = np.array(range(2004, 2024))
X_future = X_future.reshape(-1, 1)
y_future = reg.predict(X_future)
plt.plot(X_future, y_future)
plt.show()