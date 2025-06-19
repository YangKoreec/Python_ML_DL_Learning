import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    # titanic = pd.read_csv('https://github.com/Ultraluxe25/Karpov-Stepik-Introduction-to-DS-and-ML/raw/main/csv/titanic.csv')
    titanic_data = pd.read_csv('https://github.com/agconti/kaggle-titanic/raw/master/data/train.csv')
    # test = pd.read_csv('https://github.com/agconti/kaggle-titanic/raw/master/data/test.csv')

    # Готовим данные
    x = titanic_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], axis=1) # Создаем переменную с данными, на которых будет обучаться дерево
    y = titanic_data['Survived'] # Создаем переменную значения, которой будут предсказываться нашим деревом
    x = pd.get_dummies(x) # Избавляемся от строковых идентификаторов
    x = x.fillna({'Age': x['Age'].median()}) # Избавляемся от пропусков в столбце Age

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)  # Создаем выборки для обучения и теста

    clf = tree.DecisionTreeClassifier(criterion='entropy') # Создаем неограниченное дерево
    clf.fit(x_train, y_train) # Обучаем неограниченное дерево
    # tree.plot_tree(clf) # Визуализация переобученного дерева
    # plt.show() # Вывод переобученного дерева в виде графика
    # Вывод точности переобученного дерева
    print(f'Точность переобученного дерева на train выборке: {clf.score(x_train, y_train)}') # Вывод точности на обучающей выборке
    print(f'Точность переобученного дерева на test выборке: {clf.score(x_test, y_test)}') # Вывод точности на тестовой выборке
    print() # Разделительный print
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3) # Создаем ограниченное дерево (глубина 5)
    clf.fit(x_train, y_train) # Обучаем ограниченное дерево
    # Вывод точности ограниченного дерева
    print(f'Точность ограниченного дерева на train выборке: {clf.score(x_train, y_train)}')  # Вывод точности на обучающей выборке
    print(f'Точность ограниченного дерева на test выборке: {clf.score(x_test, y_test)}')  # Вывод точности на тестовой выборке