import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, roc_curve, auc


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

def auto_find_optimal_param(x : pd.DataFrame, y : pd.Series) -> None:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=52)

    # Создаем пустой классификатор
    clf = tree.DecisionTreeClassifier()

    # Создаем словарь с подбираемыми параметрами
    parametrs = {'criterion': ['gini', 'entropy'], 'max_depth': range(1, 30)}

    # Создаем классификатор GridSearchCV
    grid_search_cv_clf = GridSearchCV(clf, parametrs, cv=5)

    # Процесс подбора идеальных параметров на тренировочно множестве
    grid_search_cv_clf.fit(x_train, y_train)
    # Вывод найденных идеальных параметров
    print(f'best_params = {grid_search_cv_clf.best_params_}')
    print()

    # Сохраняем найденный лучший классификатор
    best_clf = grid_search_cv_clf.best_estimator_
    print(f'score = {best_clf.score(x_test, y_test)}')
    print()

    y_pred = best_clf.predict(x_test)
    print(f'precision = {precision_score(y_test, y_pred)}')
    print(f'recall = {recall_score(y_test, y_pred)}')
    print()

    # Что храниться внутри y_pred
    y_pred_proba = best_clf.predict_proba(x_test)
    # Гистограмма распределения предсказанных единиц
    # pd.Series(y_pred_proba[:, 1]).hist()
    # plt.show()

    # Ужесточаем критерий выбора единицы в предсказаниях
    y_pred = np.where(y_pred_proba[:, 1] > 0.8, 1, 0)
    # Новое значение precision после ужесточения
    print(f'precision (y > 0.8) = {precision_score(y_test, y_pred)}')
    # Новое значение recall после ужесточения
    print(f'recall (y > 0.8) = {recall_score(y_test, y_pred)}')
    print()

    # Смягчение критерия выбора единицы в предсказании
    y_pred = np.where(y_pred_proba[:, 1] > 0.2, 1, 0)
    # Новое значение precision после смягчения
    print(f'precision (y > 0.2) = {precision_score(y_test, y_pred)}')
    # Новое значение recall после смягчения
    print(f'recall (y > 0.2) = {recall_score(y_test, y_pred)}')
    print()

    # Построение ROC-кривой
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)

    # Создание нового графического окна
    plt.figure()

    # Построение ROC-кривой
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    # Отображение графика
    plt.show()

if __name__ == '__main__':
    titanic_data = pd.read_csv('C:/Users/Sergey/Desktop/PythonProject/DataSets/titanic.csv')

    # Готовим датасет
    x = titanic_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], axis=1)
    y = titanic_data['Survived']
    x = pd.get_dummies(x)
    x = x.fillna({'Age': x['Age'].median()})

    auto_find_optimal_param(x, y)
