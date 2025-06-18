import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree


data = pd.DataFrame({'X_1': [1, 1, 1, 0, 0, 0, 0, 1],
                     'X_2': [0, 0, 0, 1, 0, 0, 0, 1],
                     'Y': [1, 1, 1, 1, 0, 0, 0, 0]})

# Мое первое дерево
def first_tree():
    # Создание дерева
    clf = tree.DecisionTreeClassifier(criterion='entropy')

    # Вывод информации об дереве
    print(clf.__dict__)

    # Создание двух переменных для обучения дерева
    # В x хранятся данные для обучения
    # В y хранятся данные, которые будут предсказываться
    x = data[['X_1', 'X_2']]
    y = data['Y']

    # Обучение дерева
    clf.fit(x, y)

    # Вывод информации об обученном дереве
    print(clf.__dict__)

if __name__ == "__main__":
    first_tree()
