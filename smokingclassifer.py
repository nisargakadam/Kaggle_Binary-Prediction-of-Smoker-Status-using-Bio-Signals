
pip install catboost

import catboost
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd

# Load the data
train = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')

X_train = train.drop('smoking', axis=1)
y_train = train['smoking']

# Create a CatBoost pool for efficient data handling
train_pool = Pool(data=X_train, label=y_train)

# Define the parameter distributions for random search
param_dist = {
    'learning_rate': [0.01, 0.1, 0.2],
    'depth': [4, 6, 8],
    'iterations': [100, 200, 300],
    'l2_leaf_reg': [1, 3, 5],
    'rsm': [0.8, 0.9, 1.0]
}

# Initialize the CatBoostRegressor
cat_model = CatBoostClassifier()

# Perform random search
random_search = RandomizedSearchCV(estimator=cat_model, param_distributions=param_dist, n_iter=10, cv=3, scoring='neg_mean_squared_error', random_state=42)
random_search.fit(X_train, y_train)

# Display the best hyperparameters
best_params = random_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the final model with the best hyperparameters
final_model = CatBoostClassifier(**best_params)
final_model.fit(X_train, y_train)

# Make probability predictions on the X_test
y_prob = final_model.predict_proba(X_test)[:, 1]

#saving predictions

sub = pd.DataFrame()
sub['id'] = X_test['id']
sub['smoking'] = y_prob
sub.to_csv('sub_smoking_grad_cat.csv', index=False)
