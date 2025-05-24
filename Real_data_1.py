import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('event_data_train.zip')

df['date'] = pd.to_datetime(df.timestamp, unit='s')
df['day'] = df.date.dt.date
df['mark'] = df.action.where(cond=lambda x:x == 'passed', other=0).mask(cond=lambda x: x == 'passed', other=1).astype(int)

# График появления новых пользователей
def new_users():
    new_blood = df.groupby('user_id', as_index=False).day.max()
    sns.set(rc={'figure.figsize': (9, 6)})
    plt.plot(new_blood.groupby('day').user_id.nunique())
    plt.tight_layout()
    plt.show()

# Подсчет баллов каждого пользователя
def all_marks():
    users_marks = df.groupby('user_id', as_index=False).mark.sum()
    sns.histplot(users_marks.mark)
    plt.show()

all_marks()