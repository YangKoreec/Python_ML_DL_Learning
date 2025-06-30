import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

events_data = pd.read_csv('event_data_train.zip')
submissions_data = pd.read_csv('submissions_data_train.zip')

# Добавляем поля date и day в events_data
events_data['date'] = pd.to_datetime(events_data.timestamp, unit='s')
events_data['day'] = events_data.date.dt.date

# Добавляем поля date и day в submissions_data
submissions_data['date'] = pd.to_datetime(submissions_data.timestamp, unit='s')
submissions_data['day'] = submissions_data.date.dt.date
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

    # Вычисляем первый раз, когда пользователь зашел на курс
    user_min_time = events_data.groupby('user_id', as_index=False) \
        .agg({'timestamp': 'min'}) \
        .rename({'timestamp': 'min_timestamp'}, axis='columns')
    users_data = users_data.merge(user_min_time, on='user_id', how='outer')

    # Не надо так делать, это очень долго
    # events_data_train = pd.DataFrame()
    # for user_id in users_data.user_id:
    #     min_user_time = users_data[users_data['user_id'] == user_id]['min_timestamp'].item()
    #     time_threshold = min_user_time + 3 * 24 * 60 * 60
    #     user_events_data = events_data[(events_data['user_id'] == user_id) & (events_data['timestamp'] < time_threshold)]
    #     event_data_train = pd.concat([event_data_train, user_events_data])

    # Количество секунд в трёх днях
    learning_time_threshold = 3 * 24 * 60 * 60

    # Находим все действия, которые пользователь сделал за 3 дня, после своего начала курса
    events_data_train = events_data.merge(user_min_time, on='user_id', how='outer') \
    .query('(timestamp - min_timestamp) <= @learning_time_threshold')

    # Проверяем найденные значение на соответствие условию поиска
    # print(events_data_train.groupby('user_id').day.nunique().max())

    # Повторяем все действия с submissions_data
    submissions_data_train = submissions_data.merge(user_min_time, on='user_id', how='outer') \
    .query('(timestamp - min_timestamp) <= @learning_time_threshold')
    # print(submissions_data_train.groupby('user_id').day.nunique().max())

    # Готовим данные для обучения модели
    X = submissions_data_train.groupby('user_id') \
        .day.nunique().to_frame().reset_index() \
        .rename(columns={'day': 'days'})
    steps_tried = submissions_data_train.groupby('user_id') \
        .step_id.nunique().to_frame().reset_index() \
        .rename(columns={'step_id': 'steps_tried'})
    X = X.merge(steps_tried, on='user_id', how='outer')
    X = X.merge(submissions_data_train.pivot_table(index='user_id',
                                                   columns='submission_status',
                                                   values='step_id',
                                                   aggfunc='count',
                                                   fill_value=0).reset_index())
    X['correct_ratio'] = X.correct / (X.correct + X.wrong)
    X = X.merge(events_data_train.pivot_table(index='user_id',
                                              columns='action',
                                              values='step_id',
                                              aggfunc='count',
                                              fill_value=0).reset_index()[['user_id', 'viewed']],
                on='user_id',
                how='outer'
                )
    X = X.fillna(0)
    X = X.merge(users_data[['user_id', 'passed_course', 'is_gone_user']],
                on='user_id',
                how='outer')
    X = X[X.is_gone_user | X.passed_course]
    y = X['passed_course'].map(int)
    X = X.drop(['passed_course', 'is_gone_user'], axis='columns')
    X = X.set_index(X.user_id)
    X = X.drop('user_id', axis='columns')

    return (X, y)

#   Поиск шага, которого большинство завершило курс
def you_shall_not_pass():
    users_last_time = submissions_data.groupby('user_id', as_index=False) \
        .agg({'timestamp': 'max'}) \
        .rename({'timestamp': 'last_timestamp'}, axis='columns')
    print(submissions_data.merge(users_last_time, on='user_id', how='outer') \
        .query('(timestamp == last_timestamp) & (submission_status == "wrong")') \
        ['step_id'].value_counts())

if __name__ == '__main__':
    create_users_data()