import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

events_data = pd.read_csv('event_data_train.zip')
submissions_data = pd.read_csv('submissions_data_train.zip')

# Добавляем поля date и day в events_data
events_data['date'] = pd.to_datetime(events_data.timestamp, unit='s')
events_data['day'] = events_data.date.dt.date

# Поиск id Карпова
def find_carpov_id():
    coca = submissions_data.groupby('user_id', as_index=False).agg({'step_id': 'count'}).sort_values('step_id', ascending=False) # Поиск пользователей, которые решили правильно большинство примеров (counts_of_correct_answers)
    print(coca)

# График появления новых пользователей
def news_users():
    new_blood = events_data.groupby('user_id', as_index=False).day.max()
    sns.set_theme(rc={'figure.figsize': (9, 6)})
    plt.plot(new_blood.groupby('day').user_id.nunique())
    plt.tight_layout()
    plt.show()

# Подсчет баллов каждого пользователя
def all_marks():
    events_data['mark'] = events_data.action.where(cond=lambda x: x == 'passed', other=0).mask(cond=lambda x: x == 'passed', other=1).astype(int)
    users_marks = events_data.groupby('user_id', as_index=False).mark.sum()
    sns.histplot(users_marks.mark)
    plt.show()

# Строим гистограмму временных промежутков, в которые пользователи были активны
def date_distribution():
    gap_data = events_data[['user_id', 'day', 'timestamp']].drop_duplicates(subset=['user_id', 'day']) \
        .groupby('user_id')['timestamp'].apply(list) \
        .apply(np.diff).values
    gap_data = pd.Series(np.concatenate(gap_data, axis=0)) / (24 * 60 * 60)
    gap_data[gap_data < 200].hist()
    plt.show()

# Создаем DataFrame с различной информацией о пользователях
def create_users_data():
    pd.set_option('display.max_columns', None) # Настройка для вывода всех столбцов

    # Создаем DataFrame с количеством разных ответов пользователей
    users_scores = submissions_data.pivot_table(index='user_id',
                                                columns='submission_status',
                                                values='step_id',
                                                aggfunc='count',
                                                fill_value=0).reset_index()

    # Создаем DataFrame с количеством действий пользователя
    users_events_data = events_data.pivot_table(index='user_id',
                                       columns='action',
                                       values='step_id',
                                       aggfunc='count',
                                       fill_value=0).reset_index()

    # Создаем DataFrame c уникальными днями активности пользователя
    users_days = events_data.groupby('user_id').day.nunique().to_frame().reset_index()

    # Создаем DataFrame с пользователями, которые покинули курс
    now = 1526772811
    drop_out_threshold = 24 * 60 * 60 * 30
    users_data = (events_data.groupby('user_id', as_index=False).agg({'timestamp': 'max'}) \
                  .rename(columns={'timestamp': 'last_timestamp'}))
    users_data['is_gone_user'] = (now - users_data.last_timestamp) > drop_out_threshold
    users_data = users_data.merge(users_scores, on='user_id', how='outer').fillna(0)
    users_data = users_data.merge(users_events_data, on='user_id', how='outer')
    users_data = users_data.merge(users_days, on='user_id', how='outer')
    users_data['passed_course'] = users_data.passed > 170
    print(users_data)

if __name__ == '__main__':
    create_users_data()