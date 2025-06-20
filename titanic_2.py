import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


# Обучение дерева с перебором вариантов глубины (max_depth)
def incorrect_fit(x : pd.DataFrame, y : pd.Series) -> None:
    # Разделяем данные на обучающие и тестовые
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    # Неправильное обучение модели так как мы подбираем такие параметры, чтобы получить более высокий процент точности на
    # тестовой выборки, а не пытаемся найти закономерности.
    max_depth_values = range(1, 100)
    scores_data = pd.DataFrame()
    for max_depth in max_depth_values:
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
        clf.fit(x_train, y_train)
        train_score = clf.score(x_train, y_train)
        test_score = clf.score(x_test, y_test)

        temp_score_data = pd.DataFrame({'max_depth': [max_depth],
                                        'train_score': [train_score],
                                        'test_score': [test_score]})

        scores_data = pd.concat([scores_data, temp_score_data])

    scores_data_long = pd.melt(scores_data, id_vars=['max_depth'], value_vars=['train_score', 'test_score'],
                               var_name='set_type', value_name='score')
    sns.lineplot(x='max_depth', y='score', hue='set_type', data=scores_data_long)
    plt.show()

# Обучение дерева с кросвалидацией
def validation_fit(x : pd.DataFrame, y : pd.Series) -> None:
    # Разделяем данные на обучающие и тестовые
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    max_depth_values = range(1, 100)
    scores_data = pd.DataFrame()
    for max_depth in max_depth_values:
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
        clf.fit(x_train, y_train)
        train_score = clf.score(x_train, y_train)
        test_score = clf.score(x_test, y_test)
        mean_cross_val_score = cross_val_score(clf, x_train, y_train, cv=5).mean()

        temp_score_data = pd.DataFrame({'max_depth': [max_depth],
                                        'train_score': [train_score],
                                        'test_score': [test_score],
                                        'cross_val_score': [mean_cross_val_score]})

        scores_data = pd.concat([scores_data, temp_score_data])

    scores_data_long = pd.melt(scores_data,
                               id_vars=['max_depth'],
                               value_vars=['train_score', 'test_score', 'cross_val_score'],
                               var_name='set_type',
                               value_name='score')

    sns.lineplot(x='max_depth', y='score', hue='set_type', data=scores_data_long)
    plt.show()

if __name__ == '__main__':
    titanic_data = pd.read_csv('C:/Users/Sergey/Desktop/PythonProject/titanic.csv')

    # Готовим датасет
    x = titanic_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], axis=1)
    y = titanic_data['Survived']
    x = pd.get_dummies(x)
    x = x.fillna({'Age': x['Age'].median()})

    validation_fit(x, y)
