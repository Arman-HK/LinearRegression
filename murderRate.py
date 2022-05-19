import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("murderrate.csv")
murder_income = df.groupby("income").murder_year.mean().reset_index()

x = murder_income["income"]
x = x.values.reshape(-1, 1)
y = murder_income["murder_year"]
plt.scatter(x, y)

reg = linear_model.LinearRegression()
reg.fit(x, y)
y_predict = reg.predict(x)
plt.plot(x, y_predict)

x_future = np.array(range(800000, 10000000))
x_future = x_future.reshape(-1, 1)
y_future = reg.predict(x_future)
plt.plot(x_future, y_future)

plt.show()