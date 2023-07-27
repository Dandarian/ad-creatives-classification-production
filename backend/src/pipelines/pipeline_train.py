"""
Модуль: Сборный конвейер для тренировки модели.
"""

# Стандартные библиотеки.
import os

# Сторонние библиотеки.
import yaml

# Локальные модули.
from .. import data
from ..train.metrics import create_dict_metrics, save_metrics
from ..train.train import study_with_optuna, train_on_best_params
from .pipeline_preprocess import pipeline_preprocess


def pipeline_train(config_path: str):
    """
    Выполняет полный цикл тренировки модели машинного обучения.

    Загружает конфигурационные параметры из YAML файла, подгружает
    данные, разбивает их на тренировочный, валидационный и тестовый
    наборы, выполняет оптимизацию гиперпараметров с помощью библиотеки
    Optuna, обучает модель с использованием лучших гиперпараметров и
    вычисляет и сохраняет метрики эффективности модели на тестовом
    наборе данных.

    Параметры:
        @param config_path: Путь к файлу с конфигурационными
            параметрами.

    Возвращает:
        @return: None. Сохраняет результаты полного цикла тренировки в
            соответствующие файлы.
    """
    # Получение конфигов.
    with open(config_path, encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    preprocessing_config = config["preprocessing"]
    train_config = config["train"]

    # Загрузка данных.
    raw_df = data.load(train_config["raw_df_path"])
    num_df = pipeline_preprocess(
        data_frame=raw_df,
        flag_evaluate=False,
        **preprocessing_config,
        **train_config,
    )

    # Разбивка данных.
    train_set, val_set, test_set = data.split.train_val_test(
        X=num_df,
        y=raw_df[preprocessing_config["target_column"]].values,
        **preprocessing_config
    )

    # Нахождение оптимальных параметров.
    study_svc = study_with_optuna(
        train_set, val_set, **preprocessing_config, **train_config
    )

    # Тренировка модели на оптимальных параметрах.
    model = train_on_best_params(
        study_svc.best_params,
        train_set,
        **preprocessing_config,
        **train_config
    )

    # Подсчёт и сохранение метрик.
    result_metrics = create_dict_metrics(
        y_test=test_set[1],
        y_pred=model.predict(test_set[0]),
        y_proba=model.predict_proba(test_set[0]),
    )
    save_metrics(result_metrics, os.path.join(train_config["metrics_path"]))
