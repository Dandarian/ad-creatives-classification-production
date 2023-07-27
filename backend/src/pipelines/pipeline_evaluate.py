"""
Модуль: Получение предсказания на основе обученной модели.
"""

# Стандартные библиотеки.
import os

# Сторонние библиотеки.
import yaml
import joblib
import pandas as pd

# Локальные модули.
from ..data.load_save import load
from ..pipelines.pipeline_preprocess import pipeline_preprocess


def pipeline_evaluate(
    config_path: str, data_frame: pd.DataFrame = None, df_path: str = None
) -> list[float]:
    """
    Получает предсказания для рабочих данных.

    Загружает файл конфигурации, рабочие данные, обученную модель.
    Отправляет данные на предобработку. Делает предсказания на основе
    предварительно обработанных данных и сохраняет предсказания в
    CSV-файл.

    Параметры:
        @param config_path: Путь до конфигурационного файла.
        @param data_frame: DataFrame с рабочими данными (опциональный).
        @param df_path: Путь до файла с рабочими данными (опциональный).

    Возвращает:
        @return: Предсказания.
    """
    # Загрузка конфигурации из файла.
    # config_path = '../config/params.yml'
    with open(config_path, encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    preprocessing_config = config["preprocessing"]
    train_config = config["train"]
    evaluate_config = config["evaluate"]

    # Загрузка и предобработка данных.

    # Если датафрейм передан явно.
    if data_frame is not None:
        data_eval_origin = data_frame
    # Если путь до датафрейма передан явно.
    elif df_path is not None:
        data_eval_origin = load(df_path)
    else:
        data_eval_origin = load(
            df_path=os.path.join(evaluate_config["raw_df_path"])
        )

    data_eval_num = pipeline_preprocess(
        data_frame=data_eval_origin,
        flag_evaluate=True,
        **preprocessing_config,
        **evaluate_config
    )

    # Загрузка обученной модели и выполнение предсказаний.
    model = joblib.load(os.path.join(train_config["model_path"]))

    prediction = model.predict(data_eval_num).tolist()

    pd.Series(prediction).to_csv(
        evaluate_config["prediction_path"], index=False
    )

    return prediction
