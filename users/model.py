import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('data/users/data.csv')

X = data.drop("label", axis=1)
y = data["label"]

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=45)

# param_grid = {
#     'n_estimators': [50, 75, 100, 125, 159, 175, 200],
#     'max_depth': [None, 10, 15, 20, 25],
#     'min_samples_split': [2, 5, 7, 9],
#     'min_samples_leaf': [1, 2],
#     'bootstrap': [True, False]
# }

# random_search = RandomizedSearchCV(RandomForestClassifier(), param_distributions=param_grid, cv=5)
# random_search.fit(x_train, y_train)
# print(random_search.best_estimator_)

model = RandomForestClassifier(bootstrap=False, max_depth=25, min_samples_leaf=2,
                       n_estimators=159)

model.fit(x_train, y_train)

y_pred  = model.predict(x_test)

print(classification_report(y_test, y_pred))

print(accuracy_score(y_test, y_pred))

pickle.dump(model, open("users/model.pkl", 'wb'))