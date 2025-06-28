import pandas as pd
import sklearn.metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_iris


def irises():
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    predicted = dt.predict(X_test)
    print(dt.predict_proba(X_test))

def best_irises():
    iris = load_iris()
    X = iris.data
    y = iris.target

    parameters = {'max_depth': range(1, 11),
                  'min_samples_split': range(2, 11),
                  'min_samples_leaf': range(1, 11)}
    search = GridSearchCV(DecisionTreeClassifier(), parameters)
    search.fit(X, y)
    best_tree = search.best_estimator_

def best_irises_random():
    iris = load_iris()
    X = iris.data
    y = iris.target

    parameters = {'max_depth': range(1, 11),
                  'min_samples_split': range(2, 11),
                  'min_samples_leaf': range(1, 11)}
    search = RandomizedSearchCV(DecisionTreeClassifier(), parameters)
    search.fit(X, y)
    best_tree = search.best_estimator_

def best_irises_predictions():
    X_train = train.drop(columns='y')
    y_train = train['y']

    parameters = {'max_depth': range(1, 11),
                  'min_samples_split': range(2, 11),
                  'min_samples_leaf': range(1, 11)}
    search = GridSearchCV(DecisionTreeClassifier(), parameters)
    search.fit(X_train, y_train)
    best_tree = search.best_estimator_
    predictions = best_tree.predict(test)

if __name__ == '__main__':
    irises()