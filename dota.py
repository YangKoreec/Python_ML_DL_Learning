import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def normalization(string : str) -> list:
    return list(string.lstrip('[').rstrip(']').replace("'", "").split(', '))

data = pd.read_csv('https://stepik.org/media/attachments/course/4852/dota_hero_stats.csv')
data['new_roles'] = [normalization(string) for string in data.loc[:, 'roles'].values]
data = data.drop('roles', axis=1)

roles_set = set()
for i in data.loc[:, 'new_roles'].values:
    roles_set |= set(i)

plot_data_dict = {i: 0 for i in roles_set}
for i in data.loc[:, 'new_roles'].values:
    for j in plot_data_dict:
        if j in i:
            plot_data_dict[j] += 1

plot_data = pd.DataFrame({'roles': plot_data_dict.keys(), 'counts': plot_data_dict.values()}, index=range(len(plot_data_dict)))
print(plot_data)
plt.bar_label(sns.histplot(plot_data, x='roles', weights='counts').containers[0])
plt.tight_layout()
plt.show()
