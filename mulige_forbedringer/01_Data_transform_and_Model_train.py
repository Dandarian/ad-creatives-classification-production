"""
Просто файл сгенерированный из ноутбука чтобы можно было брать из него
методы и другие куски кода.
"""
#!/usr/bin/env python
# coding: utf-8
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, log_loss
import optuna
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization import plot_optimization_history
from umap import UMAP
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.axes as axes
import pymorphy2
import yaml
import json
import joblib

# In[2]:

config_path = os.path.join('../config/params.yml')
config = yaml.load(open(config_path), Loader=yaml.FullLoader)


preproc = config['preprocessing']
training = config['train']


# In[7]:


RAND = preproc["random_state"]
target_column = preproc["target_column"]
target_column_pred = preproc["target_column_pred"]


# ## Загрузка датасета

# In[9]:


df_plus_cats = pd.read_csv(
    preproc['train_path'], index_col=0, keep_default_na=False)


# In[17]:



# In[22]:


# Создание числовых признаков


# Целевая функция
#


plotly_config = {"staticPlot": True}

fig = plot_optimization_history(study_svc)
fig.show(config=plotly_config)


# In[10]:


plot_param_importances(study_svc);





# In[69]:





# In[70]:



# In[71]:



# Значение на тестовой выборке получилось выше, чем до тюнинга

# Далее я также отдельно попробовал:
# - Применял масштабирование при помощи StandardScaler
# - Обучал на других ядрах при прочих дефолтных параметрах  
# - Увеличивал С и гамма  
# - Пробовал полиномы с большими степенями и средними значениями гамма и С

# Но результат первой подборки оказался всё равно лучше

# ## Predict

# In[77]:


numbers_words[target_column_pred] = svc_optuna.predict(numbers_words)


# In[78]:


numbers_words


# In[92]:


def get_df_predict_origin(df_predict: pd.DataFrame,
                          df_origin: pd.DataFrame,
                          drop_columns: list,
                          rename_columns: dict) -> pd.DataFrame:
    """
    Соединяет в один датафрейм предсказанные категории и изначальные
    :param df_predict: датафрейм с признаками и предсказанными значениями целевой
        переменной -- категории
    :param df_origin: датафрейм с признаками и изначальными значениями категории
    :param drop_columns: колонки для удаления
    :param rename_columns: колонки для переименования
    :return: датафрейм с признаками и предсказанными и изначальными значениями категории
    """
    df_origin = df_origin.reset_index(drop=True)
    # Соединяем только одну колонку cat_pred со всем датасетом df_origin
    df = pd.concat([df_predict[target_column_pred],
                    df_origin],
                   axis=1)
    # Удаляем ненужные и переименовываем столбцы
    df.drop(drop_columns, axis=1, inplace=True, errors='ignore')
    df.rename(rename_columns, axis=1, inplace=True, errors='ignore')
    
    return df


# In[93]:


df_predict_origin = get_df_predict_origin(numbers_words,
                                          df_plus_cats,
                                          preproc['drop_columns'],
                                          preproc['rename_columns'])
df_predict_origin


# In[94]:


if target_column in preproc['rename_columns']:
    target_column = preproc['rename_columns'][target_column]
if target_column_pred in preproc['rename_columns']:
    target_column_pred = preproc['rename_columns'][target_column_pred]


# In[82]:


barplot_category_percents(df_predict_origin[target_column_pred],
                          'Количество объектов в предсказанной категории')


# Видим, что распределение в целом похожее на то, которое делали для изначальных категорий

# In[85]:


def embeddings_diagram(X: pd.DataFrame, X_name: str) -> None:
    """
    Принимает набор данных. Создаёт эмбеддинги сниженной размерности.
    Предсказывает для него значения целевой переменной. 
    Рисует диаграмму рассеяния
    :param X: набор данных
    :param X_name: имя набора данных
    :return: None, значения не возвращает, но итогом работы функции является
        отображение диаграммы
    """
    um = UMAP(n_components=3, random_state=RAND, n_neighbors=15, min_dist=0.1)
    X_embedding = um.fit_transform(X)

    y_pred = svc_optuna.predict(X)

    plt.figure(figsize=(13, 8))
    plt.title(f'Диаграмма рассеяния предсказанных значений целевой переменной '
              f'на наборе данных: {X_name}',
              fontsize=14)
    sns.scatterplot(x=X_embedding[:, 0],
                    y=X_embedding[:, 1],
                    hue=y_pred,
                    s=100,
                    legend='full')


# In[86]:


embeddings_diagram(X_train, "Обучающий")


# Видны территории категорий со своими границами, с некоторыми выбросами.

# In[87]:


embeddings_diagram(X_test, "Тестовый")


# На тесте разделение есть, но категории расположены в других местах.  
# (Алгоритм не гарантирует что положение будет сохранено,  
# т.к. задача нахожождения наилучшей проекции имеет много решений)  
# Главное, что категории группируются, это хорошо

# Посмотрим визуально датасет.

# In[95]:


df_predict_origin[2000:2010]


# In[96]:


df_predict_origin[df_predict_origin[target_column_pred]
    != df_predict_origin[target_column]].shape


# Количество отличающихся предсказанных и изначальных значений составляет примерно 4%

# После визуального осмотра можно сказать, что среди различающихся значений модель где-то вернее предсказала категорию, где-то нет.
# Но в общем и целом показала результат не хуже, чем изначальная разметка.
