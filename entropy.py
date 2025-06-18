import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
from sklearn import tree


# Функция для вычисления энтропии элементов объекта тип Series в бинарной классификации
# ser - передаваемый объект типа pd.Series
def find_entropy(ser : pd.Series) -> float:
    # Общее количество элементов в ser
    n = ser.size
    # Количество элементов равных нулю
    count_0 = ser.loc[ser == 0].size
    # Количество элементов равных единице
    count_1 = ser.loc[ser == 1].size

    # Вычисление вероятности p для нулей и для единиц
    if n != 0:
        p_0 = count_0 / n
        p_1 = count_1 / n
    else:
        p_0, p_1 = 0, 0

    # Вычисление логарифмов для подсчета энтропии
    log2_0 = 0 if p_0 == 0 else math.log2(p_0)
    log2_1 = 0 if p_1 == 0 else math.log2(p_1)

    # Вычисление энтропии
    entropy = - (p_1 * log2_1 + p_0 * log2_0)

    return entropy

if __name__ == '__main__':
    data = pd.read_csv('https://stepik.org/media/attachments/course/4852/cats.csv')
    data.drop('Unnamed: 0', inplace=True, axis=1)

    data['Вид'] = data.replace({'собачка': 0, 'котик': 1})['Вид'].astype(np.int64)

    entropy = find_entropy(data['Вид'])
    n = data['Вид'].size
    for column in ['Шерстист', 'Гавкает', 'Лазает по деревьям']:
        cond_0 = data[data[column] == 0]['Вид']
        cond_1 = data[data[column] == 1]['Вид']
        n1_n = cond_0.size / n
        n2_n = cond_1.size / n
        print(f'IG для {column} = {round(entropy - (n1_n * find_entropy(cond_0) + n2_n * find_entropy(cond_1)), 2)}')