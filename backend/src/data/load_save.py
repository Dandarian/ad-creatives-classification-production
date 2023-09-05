"""
Модуль: Загрузка и сохранение датафреймов.
"""

# Стандартные библиотеки.
from typing import Text

# Сторонние библиотеки.
import pandas as pd


def load(df_path: Text) -> pd.DataFrame:
    """
    Загружает датафрейм из CSV-файла по заданному пути.

    Параметры:
        @param df_path: Путь к CSV-файлу.

    Возвращает:
        @return: Датафрейм.
    """
    return pd.read_csv(df_path, index_col=0, keep_default_na=False)


def save(data_frame: pd.DataFrame, df_path: Text) -> None:
    """
    Сохраняет датафрейм в CSV-файл по заданному пути.

    Параметры:
        @param data_frame: Датафрейм.
        @param df_path: Путь до файла.

    Возвращает:
        @return: None
    """
    data_frame.to_csv(df_path)
