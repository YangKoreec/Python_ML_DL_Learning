import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('https://stepik.org/media/attachments/course/4852/iris.csv')
data = data.drop('Unnamed: 0', axis=1)
sns.violinplot(data.loc[:, 'petal length'])
plt.show()