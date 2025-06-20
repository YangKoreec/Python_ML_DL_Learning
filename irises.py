import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


def main():
    irises_train = pd.read_csv('https://stepik.org/media/attachments/course/4852/train_iris.csv', index_col=0)
    irises_test = pd.read_csv('https://stepik.org/media/attachments/course/4852/test_iris.csv', index_col=0)

    X_train = irises_train.drop('species', axis=1)
    y_train = irises_train['species']

    X_test = irises_test.drop('species', axis=1)
    y_test = irises_test['species']

    all_scores = pd.DataFrame()

    np.random.seed(0)
    for max_depth in range(1, 100):
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
        clf.fit(X_train, y_train)

        temp_train_score = clf.score(X_train, y_train)
        temp_test_score = clf.score(X_test, y_test)
        temp_scores_data = pd.DataFrame({'max_depth' : [max_depth],
                                         'train_score': [temp_train_score],
                                         'test_score': [temp_test_score]})

        all_scores = pd.concat([all_scores, temp_scores_data])

    all_scores_long = pd.melt(all_scores,
                              id_vars=['max_depth'],
                              value_vars=['train_score', 'test_score'],
                              var_name='set_type',
                              value_name='score')

    sns.lineplot(x='max_depth', y='score', hue='set_type', data=all_scores_long)
    plt.show()

if __name__ == '__main__':
    main()