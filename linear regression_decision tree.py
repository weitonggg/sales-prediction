import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Multivariate Linear Regression Model 
lr = LinearRegression()
lr.fit(x_train, y_train)

lr_preds = lr.predict(x_test)
lr_mse = mean_squared_error(lr_preds, y_test)
lr_rmse = np.sqrt(lr_mse)

# Decision Tree
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)

dt_preds = dt.predict(x_test)
dt_mse = mean_squared_error(dt_preds, y_test)
dt_rmse = np.sqrt(dt_mse)

