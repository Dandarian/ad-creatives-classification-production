"""
Модуль: что-то оставшееся от evaluate ноутбука. Куда-то можно
приспособить. Тут соединение в датасет предсказанных значений.
"""
# !/usr/bin/env python
# coding: utf-8
import warnings
import yaml
import pandas as pd
from backend.src.pipelines.pipeline_evaluate import pipeline_evaluate
# from typing import Tuple, Union, Any, Optional
from backend.src import data

warnings.filterwarnings("ignore")


def make_df_predict_origin(**kwargs) -> pd.DataFrame:
    """
    Соединяет в один датафрейм предсказанные категории и изначальные.

    Ключи-аргументы:
        @kwarg prediction_path: Путь к списку с предсказанными
            значениями целевой переменной -- категории.
        @kwarg raw_df_path: Путь до датафрейма с признаками и
            изначальными значениями категории.
        @kwarg target_column_pred: Имя колонки для предсказанных
            значений.
        @kwarg drop_columns: Колонки для удаления.
        @kwarg rename_columns: Колонки для переименования.

    Возвращает:
        @return: Датафрейм с признаками и предсказанными и изначальными
            значениями категории.
    """
    data_frame = data.load(kwargs["raw_df_path"])
    prediction = data.load(kwargs["prediction_path"]).to_list()
    # Добавляем только одну колонку cat_pred в датафрейм df_origin
    data_frame[kwargs["target_column_pred"]] = prediction
    # Удаляем ненужные и переименовываем столбцы
    data_frame.drop(
        kwargs["drop_columns"], axis=1, inplace=True, errors="ignore"
    )
    data_frame.rename(
        kwargs["rename_columns"], axis=1, inplace=True, errors="ignore"
    )

    return data_frame


# get configs
CONFIG_PATH = os.path.join("config/params.yml")
with open(CONFIG_PATH) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
preprocessing_config = config["preprocessing"]
train_config = config["train"]
evaluate_config = config["evaluate"]

data_eval_predict_origin = make_df_predict_origin(
    **preprocessing_config, **evaluate_config
)
data_eval_predict_origin


# In[ ]:


# Как видим, из 10ти значений:
# - 4 -- точно верно предсказаны
# - 4 -- пойдёт, но лучше бы подошла другая категория
# - 2 -- точно не верно предсказаны

# Чтобы оценить точность, выберем рандомно 100 строк и проверим вручную.
# Что будет грубой оценкой, но другой возможности оценить нет, поскольку
# у нас есть только предсказанные значения целевой переменной

# In[189]:


random_sample = data_eval_predict_origin.sample(n=100)


# - 55 -- точно верно предсказаны (присваиваем 1 балл)
# - 22 -- пойдёт, но лучше бы подошла другая категория (присваиваем 0.5 балла)
# - 23 -- точно не верно предсказаны (присваиваем 0 баллов)
# итого 66

# Примерная точность 66%
#
# Такое значение связано с тем, что в evaluate сете много новых
# признаков и много недостающих,
# что значит много незнакомых для модели слов
#

# In[187]:


len_cols


#
# Учитывая, что на обучающем датасете точность была 96%,
# можно сделать вывод, что в дальнейшем расширяя обучающий датасет,
# мы добьёмся большей точности и на evaluate датасете
