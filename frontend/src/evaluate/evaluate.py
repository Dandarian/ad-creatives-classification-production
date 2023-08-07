"""
Программа: Отрисовка слайдеров и кнопок для ввода данных
с дальнейшим получением предсказания на основании введенных значений
Версия: 1.0
"""

import json
import os
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import streamlit as st


def evaluate_input(endpoint: object) -> None:
    """
    Получение входных данных путем ввода в UI -> вывод результата
    :param endpoint: endpoint
    """
    # поля для вводы данных, используем уникальные значения
    title = st.sidebar.text_input("Title")
    description = st.sidebar.text_input("Description")
    adomain = st.sidebar.text_input("Adomain")
    bundle = st.sidebar.text_input("Bundle")

    dict_data = {
        "Title": title,
        "Description": description,
        "Adomain": adomain,
        "Bundle": bundle,
    }

    st.write(
        f"""### Данные креатива:\n
    1) Title: {dict_data['Title']}
    2) Description: {dict_data['Description']}
    3) Adomain: {dict_data['Adomain']}
    4) Bundle: {dict_data['Bundle']}
    """
    )

    # evaluate and return prediction (text)
    button_ok = st.button("Predict")
    if button_ok:
        result = requests.post(endpoint, timeout=8000, json=dict_data)
        json_str = json.dumps(result.json())
        output = json.loads(json_str)
        st.write(f"## {output}")
        st.success("Success!")


def evaluate_from_file(data: pd.DataFrame, endpoint: object, files: BytesIO):
    """
    Получение входных данных в качестве файла -> вывод результата в виде таблицы
    :param data: датасет
    :param endpoint: endpoint
    :param files:
    """
    button_ok = st.button("Predict")
    if button_ok:
        # заглушка так как не выводим все предсказания
        data_ = data[:5]
        output = requests.post(endpoint, files=files, timeout=8000)
        print(output)
        data_["predict"] = output.json()["prediction"]
        st.write(data_.drop(columns=['Unnamed: 0']).head())


def show_old_pred_results(config: dict) -> None:
    """
    Вывод старых результатов предсказания.
    @param config:
    @return:
    """
    if os.path.exists(config["evaluate"]["raw_df_path"])\
            and os.path.exists(config["evaluate"]["prediction_path"]):
        data_ = pd.read_csv(
            config["evaluate"]["raw_df_path"],
            # заглушка так как не выводим все предсказания
            nrows=5,
            # index_col=0,
            keep_default_na=False
        )
        st.write("Last prediction results:")
        # Чтение CSV-файла с разделителем перехода на новую строку
        with open(config["evaluate"]["prediction_path"], 'r') as file:
            lines = file.read().splitlines()
        # Преобразование списка строк в массив NumPy
        predict_ = np.array(lines[:5], dtype=str)
        data_["predict"] = predict_
        st.write(data_.drop(columns=['Unnamed: 0']).head())
