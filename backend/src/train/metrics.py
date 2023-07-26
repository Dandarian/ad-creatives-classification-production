"""
Модуль: Получение метрик.
"""

# Стандартные библиотеки.
import json

# Сторонние библиотеки.
import pandas as pd
import yaml
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)


def create_dict_metrics(
    y_test: pd.Series,
    y_pred: pd.Series,
    y_proba: pd.Series,
    multi_class: str = "ovr",
    average: str = "weighted",
) -> dict:
    """
    Рассчитывает метрики для задачи классификации и записывает в
    словарь.

    Параметры:
        @param y_test: Реальные данные.
        @param y_pred: Предсказанные значения.
        @param y_proba: Предсказанные вероятности.
        @param multi_class: optional (по умолчанию "ovr")
            Стратегия обработки многоклассовой классификации.
            Возможные значения:
            "ovr" (один против всех),
            "ovo" (один против одного).
        @param average: optional (по умолчанию "weighted")
            Стратегия для вычисления усредненной метрики для
            многоклассовой классификации.
            Возможные значения: "micro", "macro", "weighted", "samples".

    Возвращает:
        @return: Словарь с метриками.
    """
    dict_metrics = {
        "roc_auc": round(
            roc_auc_score(y_test, y_proba, multi_class=multi_class), 3
        ),
        "precision": round(
            precision_score(y_test, y_pred, average=average), 3
        ),
        "recall": round(recall_score(y_test, y_pred, average=average), 3),
        "f1": round(f1_score(y_test, y_pred, average=average), 3),
        "logloss": round(log_loss(y_test, y_proba), 3),
    }

    return dict_metrics


def save_metrics(result_metrics, metric_path: str) -> None:
    """
    Сохраняет метрики.

    Параметры:
        @param result_metrics: Словарь с рассчитанными метриками.
        @param metric_path: Путь для сохранения метрик.

    Возвращает:
        @return: None.
    """
    with open(metric_path, "w", encoding='utf-8') as file:
        json.dump(result_metrics, file)


def load_metrics(config_path: str) -> dict:
    """
    Получает метрики из файла.

    Параметры:
        @param config_path: Путь до файла с конфигом.

    Возвращает:
        @return: Словарь с рассчитанными метриками.
    """
    with open(config_path, encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    train_config = config["train"]

    with open(train_config["metrics_path"], encoding='utf-8') as json_file:
        metrics = json.load(json_file)

    return metrics
