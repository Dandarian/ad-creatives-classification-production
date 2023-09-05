"""
Модуль: Получение данных по пути и чтение.
"""

from io import BytesIO
import io
from typing import Dict, Tuple
import streamlit as st
import pandas as pd


def get_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Получение данных по заданному пути
    :param dataset_path: путь до данных
    :return: датасет
    """
    return pd.read_csv(dataset_path, keep_default_na=False)


def load_data(
    data: str, type_data: str
) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, BytesIO, str]]]:
    """
    Получение данных и преобразование в тип BytesIO для обработки в streamlit
    :param data: данные
    :param type_data: тип датасет (train/test)
    :return: датасет, датасет в формате BytesIO
    """
    # Не удаляем индекс, т. к. передаём на бэкенд и индекс будет
    # удаляться там: index_col=0, keep_default_na=False.
    dataset = pd.read_csv(data)
    st.write("Dataset load")
    # Убираем индекс только для отображения.
    st.write(dataset.drop(columns=['Unnamed: 0']).head())

    # Преобразовать dataframe в объект BytesIO (для последующего анализа в виде файла в FastAPI)
    dataset_bytes_obj = io.BytesIO()
    # запись в BytesIO буфер
    dataset.to_csv(dataset_bytes_obj, index=False)
    # Сбросить указатель, чтобы избежать ошибки с пустыми данными
    dataset_bytes_obj.seek(0)

    files = {
        "file": (f"{type_data}_dataset.csv", dataset_bytes_obj, "multipart/form-data")
    }
    return dataset, files
