import pandas as pd

my_stat = pd.read_csv('https://stepik.org/media/attachments/course/4852/my_stat_1.csv')

my_stat.fillna({'session_value': 0}, inplace=True)
med = my_stat.query("n_users > 0").loc[:, 'n_users'].median()
my_stat['n_users'] = my_stat['n_users'].replace(my_stat.query('n_users < 0').loc[:, 'n_users'].values, med)