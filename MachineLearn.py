import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

my_stat = pd.read_csv('https://stepik.org/media/attachments/course/4852/my_stat.csv')

subset_1 = my_stat.iloc[0:10, [0, 2]]
subset_2 = my_stat.iloc[:, [1, 3]].drop([0, 4], axis=0)

my_stat.rename({'V1': 'session_value', 'V2': 'group', 'V3': 'time', 'V4': 'n_users'}, inplace=True, axis=1)