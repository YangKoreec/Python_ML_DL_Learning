import pandas as pd

my_stat = pd.read_csv('https://stepik.org/media/attachments/course/4852/my_stat_1.csv')
my_stat.fillna(0, inplace=True)
print(my_stat)
agg_args = {'session_value': 'mean'}
rename_arg = {'session_value': 'mean_session_values'}
group = my_stat.groupby('group', as_index=False)
mean_session_value_data = group.agg(agg_args).rename(rename_arg, axis=1)
print(mean_session_value_data)