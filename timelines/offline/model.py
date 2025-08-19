import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('data/tweets/tweets_data.csv')

X = data.drop("label", axis=1)
y = data["label"]

X.loc[:,"time_of_day"] = X["time_of_day"].map({"morning":0, "evening":1, "afternoon":2, "evening":3, "night":4})

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=45)

# param_grid = {
#     'n_estimators': [50, 75, 100, 125],
#     'max_depth': [None, 15, 20, 25],
#     'min_samples_split': [2, 5, 7],
#     'min_samples_leaf': [1, 2],
#     'bootstrap': [True, False]
# }

# random_search = RandomizedSearchCV(RandomForestClassifier(), param_distributions=param_grid, cv=5)
# print('-')
# random_search.fit(x_train, y_train)
# print(random_search.best_estimator_)

model = RandomForestClassifier(bootstrap=False, max_depth=25, min_samples_split=7)

model.fit(x_train, y_train)

y_pred  = model.predict(x_test)

print(classification_report(y_test, y_pred))

print(accuracy_score(y_test, y_pred))

pickle.dump(model, open("timelines/offline/model.pkl", 'wb'))