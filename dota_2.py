import pandas as pd
import matplotlib.pyplot as plt

def normalization(string : str) -> list:
    return list(string.lstrip('[').rstrip(']').replace("'", "").split(', '))

data = pd.read_csv('https://stepik.org/media/attachments/course/4852/dota_hero_stats.csv')
data['new_roles'] = [normalization(i) for i in data.loc[:, 'roles'].values]
data = data.drop(['Unnamed: 0', 'roles'], axis=1)
data['role_count'] = [len(i) for i in data.new_roles.values]
plt.figure(figsize=(12, 5))
plt.hist(range(data.name.size), weights=data.role_count, bins=data.role_count.size)
plt.tight_layout()
plt.show()

print(data.loc[:, 'role_count'].mode()[0])