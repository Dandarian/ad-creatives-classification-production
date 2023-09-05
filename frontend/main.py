"""
Главный файл: Frontend часть проекта.
"""
import json
import os

import joblib
import pandas as pd
import numpy as np
import yaml
import streamlit as st
from src.data.get_data import load_data, get_dataset
from src.visualization.charts import words_count_bars, \
    barplot_category_percents, embeddings_diagram, \
    total_len_words
from src.train.training import start_training, show_old_results
from src.evaluate.evaluate import evaluate_input, evaluate_from_file, \
    show_old_pred_results

CONFIG_PATH = "../config/params.yml"


def main_page():
    """
    Страница с описанием проекта.
    """
    st.markdown("# Классификатор по категориям рекламных "
                "роликов на основе их текстовых описаний.")

    # name of the columns
    st.markdown(
        """
        ### Описание текстовых полей рекламного креатива:
            - title - заголовок рекламаного креатива, обычно это название 
                      рекламируемого продукта.
            - description - описание рекламного ролика, более подробная 
                            информация о ролике и продукте.
            - adomain - домен рекламодателя, зачастую содержит название 
                        продукта.
            - bundle - наименование бандла, также может содержать название 
                       продукта.
            - cat - наименование категории. Целевая переменная. (Например 
                    Food & Drinks или Hobbies)
    """
    )


def training():
    """
    Тренировка модели
    """
    st.markdown("# Training model Support Vector Classifier")
    # get params
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # endpoint
    endpoint = config["endpoints"]["train"]

    if st.button("Start new training"):
        start_training(config=config, endpoint=endpoint)
    else:
        show_old_results(config=config)


def prediction():
    """
    Получение предсказаний путем ввода данных
    """
    st.markdown("# Prediction")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["prediction_input"]

    # проверка на наличие сохраненной модели
    if os.path.exists(config["train"]["model_path"]):
        evaluate_input(endpoint=endpoint)
    else:
        st.error("Сначала обучите модель")


def prediction_from_file():
    """
    Получение предсказаний из файла с данными
    """
    st.markdown("# Prediction")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["prediction_from_file"]

    upload_file = st.file_uploader(
        "", type=["csv", "xlsx"], accept_multiple_files=False
    )
    # проверка загружен ли файл
    if upload_file:
        dataset_csv_df, files = load_data(data=upload_file, type_data="Test")
        # проверка на наличие сохраненной модели
        if os.path.exists(config["train"]["model_path"]):
            evaluate_from_file(data=dataset_csv_df, endpoint=endpoint, files=files)
        else:
            st.error("Сначала обучите модель")
    else:
        show_old_pred_results(config=config)


def exploratory():
    """
    Exploratory data analysis.
    """
    st.markdown("# Analysis & Visualization")

    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load and write datasets
    st.write("Training dataframe:")
    data = get_dataset(dataset_path=config["train"]["cng_df_path"])
    st.write(data.drop(columns=['Unnamed: 0']).head())

    st.write("Production dataframe:")
    data_prod = get_dataset(dataset_path=config["evaluate"]["cng_df_path"])
    st.write(data_prod.drop(columns=['Unnamed: 0']).head())

    # visualization with checkbox
    words_count_bars_box = st.sidebar.checkbox(
        "Распределение кол-ва слов в объектах"
    )
    barplot_category_percents_box = st.sidebar.checkbox(
        "Количество объектов в категории тренировочного датасета"
    )
    barplot_predicted_category_percents_box = st.sidebar.checkbox(
        "Количество объектов в предсказанной категории рабочего датасета"
    )
    embeddings_diagram_box = st.sidebar.checkbox(
        "Диаграмма рассеяния предсказанных значений категории рабочего "
        "датасета"
    )
    diff_words = st.sidebar.checkbox(
        "Количество новых и недостающих слов в рабочем датасете по "
        "сравнению с обучающим датасетом"
    )

    if words_count_bars_box:
        st.pyplot(
            words_count_bars(
                data_frame=data,
                col="text_count",
            )
        )
    if barplot_category_percents_box:
        st.pyplot(
            barplot_category_percents(
                sr=data[config["preprocessing"]["target_column"]],
                title="Количество объектов в категории тренировочного датасета"
            )
        )
    if barplot_predicted_category_percents_box:
        st.pyplot(
            barplot_category_percents(
                sr=pd.read_csv(
                    os.path.join(config["evaluate"]["prediction_path"]),
                    header=None,
                    squeeze=True
                )[1:],
                title="Количество объектов в предсказанной категории "
                      "рабочего датасета"
            )
        )
    if embeddings_diagram_box:
        X = pd.read_csv(
            os.path.join(config["evaluate"]["checked_num_df_path"]),
            index_col=0,
            keep_default_na=False
        )
        # Чтение CSV-файла с разделителем перехода на новую строку
        with open(config["evaluate"]["prediction_path"], 'r') as file:
            lines = file.read().splitlines()
        # Преобразование списка строк в массив NumPy
        y_pred = np.array(lines[1:], dtype=str)
        # model = joblib.load(os.path.join(config["train"]["model_path"]))

        # y_pred = model.predict(X)
        st.pyplot(
            embeddings_diagram(
                X=X,
                y_pred=y_pred,
                title="Диаграмма рассеяния предсказанных значений категории "
                      "рабочего датасета",
                random_state=config["preprocessing"]["random_state"]
            )
        )
    if diff_words:
        with open(config["evaluate"]["len_cols_path"]) as json_file:
            len_cols = json.load(json_file)

        st.write("Количество новых и недостающих слов в рабочем датасете по "
                 "сравнению с обучающим датасетом:")
        st.write("Новые слова: ", len_cols["new_columns"])
        st.write("Недостающие слова: ", len_cols["missed_columns"])

        st.write("Общее количество слов в обучающем датасете: ",
                 total_len_words(config["train"]["num_df_path"]))
        st.write("Общее количество слов в рабочем датасете: ",
                 total_len_words(config["evaluate"]["num_df_path"]))


def main():
    """
    Сборка пайплайна в одном блоке.
    """
    page_names_to_funcs = {
        "Description": main_page,
        "Training": training,
        "Prediction from file": prediction_from_file,
        "Prediction from input": prediction,
        "Analysis & Visualization": exploratory,
    }
    selected_page = st.sidebar.selectbox("Выберите пункт", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()


if __name__ == "__main__":
    main()
