"""
Модуль: Предварительная обработка данных.
"""

# Стандартные библиотеки.
import os
import warnings

# Сторонние библиотеки.
import pandas as pd

# Локальные модули.
from .. import data
from ..data.transform import (
    sum_text,
    text_count,
    stem,
    tf_idf,
    check_columns_evaluate,
)

warnings.filterwarnings("ignore")

# from typing import Tuple, Union, Any, Optional


def pipeline_preprocess(
    data_frame: pd.DataFrame, flag_evaluate: bool = True, **kwargs
) -> pd.DataFrame:
    """
    Применяет последовательность шагов для предобработки данных.

    Входной датафрейм с признаками проходит через обработчики, такие как
    добавление столбцов суммирования, подсчет текстовых данных, стемминг
    слов, преобразование в числовые признаки с помощью метода TF-IDF и
    сохранение данных в файлы.

    Параметры:
        @param data_frame: Оригинальный датафрейм с признаками.
        @param flag_evaluate: По умолчанию True. Флаг, указывающий,
            что на предобработку подаётся evaluate датафрейм и следует
            выполнить дополнительные проверки данных для оценки.
        @param kwargs: Передаются в качестве аргументов в дальнейшие
            функции.

    Возвращает:
        @return: Датафрейм, готовый к предсказанию.
    """
    # Добавление столбцов.
    sum_text(df=data_frame, sum_columns=kwargs["sum_columns"])
    text_count(df=data_frame, col_name="text")

    # Преобразование текстов.
    list_text = list(data_frame["text"].values)
    stem(lst=list_text, unnecessary_words=kwargs["unnecessary_words"])

    # Преобразование в числовые признаки с помощью TF-IDF.
    data_num = tf_idf(
        list_text=list_text,
        vectorizer_params=kwargs["vectorizer_params"],
    )
    data.save(data_num, os.path.join(kwargs["num_df_path"]))

    if flag_evaluate:
        # проверка датасета на совпадение с признаками из train
        data_num = check_columns_evaluate(data_num, **kwargs)
        data.save(data_num, os.path.join(kwargs["checked_num_df_path"]))

    return data_num
