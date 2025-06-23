import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import cross_val_score


def data_preparation(data : pd.DataFrame) -> tuple:
    data = pd.read_csv('C:/Users/Sergey/Desktop/PythonProject/dogs_n_cats.csv')
    data['Вид'] = data['Вид'].map({'собачка': 0, 'котик': 1})
    X = data.drop('Вид', axis='columns')
    y = data['Вид']

    return (X, y)

# Ищем оптимальный параметр глубины дерева
def find_optimal_depth(data : pd.DataFrame):
    X, y = data_preparation(data)

    scores_data = pd.DataFrame()

    for max_depth in range(1, 100):
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
        clf.fit(X, y)
        train_score = clf.score(X, y)
        mean_cross_val_score = cross_val_score(clf, X, y, cv=10).mean()
        scores_data = pd.concat([scores_data, pd.DataFrame({'max_depth': [max_depth],
                                                            'cross_val_score': [mean_cross_val_score],
                                                            'train_score': [train_score]})])

    scores_data_long = pd.melt(scores_data,
                               id_vars=['max_depth'],
                               value_vars=['cross_val_score', 'train_score'],
                               var_name='score_type',
                               value_name='score_value')

    sns.lineplot(x='max_depth',
                 y='score_value',
                 hue='score_type',
                 data=scores_data_long)
    plt.show()

def main():
    data = pd.read_csv('C:/Users/Sergey/Desktop/PythonProject/dogs_n_cats.csv')
    X, y = data_preparation(data)

    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
    clf.fit(X, y)

    new_data = pd.read_json('C:/Users/Sergey/Desktop/PythonProject/dataset_209691_15.json')
    answer = pd.Series(clf.predict(new_data), name='Вид')
    answer = answer.map({0: 'собачка', 1: 'котик'})
    print(answer.value_counts())

if __name__ == '__main__':
    main()