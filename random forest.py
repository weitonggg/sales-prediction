import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

rf_base = RandomForestRegressor()
rf_base.fit(x_train, y_train)


# random forest with randomized search CV
n_estimators = np.arange(200,2200,200)
max_features = ['auto', 'sqrt']
max_depth = np.arange(10,110,20)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf_random = RandomizedSearchCV(RandomForestRegressor(), 
                               random_grid, n_iter = 50, cv = 3, random_state=1)
rf_random.fit(x_train, y_train)
#rf_random.best_params_

# random forest with grid search CV
param_grid = {'bootstrap': [True],
 'max_features': [4,5,6],
 'min_samples_leaf': [3,4,5],
 'min_samples_split': [2,3,4],
 'n_estimators': [900, 1000, 1100,]}

rf_grid = GridSearchCV(RandomForestRegressor(), param_grid, cv = 3)

rf_grid.fit(x_train, y_train)
#rf_grid.best_params_


def evaluate_fit(model, x_test, y_test):
    preds = model.predict(x_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)

    return model.__class__.__name__, \
         'MSE: {}, RMSE: {}' \
         .format(mse, rmse)
         
evaluate_fit(rf_base, x_test, y_test)
evaluate_fit(rf_random.best_estimator_, x_test, y_test)
evaluate_fit(rf_grid.best_estimator_, x_test, y_test)


'''
- Random Forest Classifier (no tuning),
 'MSE: 1435006.8143441523, RMSE: 1197.9176993200126'
- Random Forest Classifier with Randomized Search,
 'MSE: 1255740.527492672, RMSE: 1120.5982899740084'
- Random Forest Classifier with Grid Search,
 'MSE: 1238100.3729259844, RMSE: 1112.6995879059111'
'''


# best params for RF using randomized search
'''
{'bootstrap': True,
 'max_features': 'sqrt',
 'min_samples_leaf': 4,
 'min_samples_split': 2,
 'n_estimators': 1000}
'''
# best params for RF using grid search
'''
{'bootstrap': True,
 'max_features': 6,
 'min_samples_leaf': 5,
 'min_samples_split': 2,
 'n_estimators': 900}
'''
