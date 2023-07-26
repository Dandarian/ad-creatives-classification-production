"""
Модуль: Тренировка модели.
"""

# Стандартные библиотеки.
import json
import os
from typing import Tuple

# Сторонние библиотеки.
import joblib
import optuna
import pandas as pd
from optuna import Study
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC


def objective_svc(
        trial: optuna.Trial,
        train_set: Tuple[pd.DataFrame, pd.DataFrame],
        val_set: Tuple[pd.DataFrame, pd.DataFrame],
        **kwargs
) -> float:
    """
    Целевая функция для оптимизации параметров модели Support Vector
    Classifier с использованием библиотеки Optuna.

    Параметры:
        @param trial: Объект, представляющий итерацию выбора
            гиперпараметров модели во время оптимизации.
        @param train_set: Кортеж из
            датасета признаков для обучения модели и
            целевых значений, соответствующих датасету признаков для
            обучения.
        @param val_set: Кортеж из
            датасета признаков для валидации модели и
            целевых значений соответствующих датасету признаков для
            валидации.

    Ключи-аргументы:
        @kwarg params_path: Путь до старых лучших параметров.
        @kwarg random_state: Число, фиксирующее начальное состояние для
            воспроизводимости результатов.

    Возвращает:
        @return: Значение ROC-AUC метрики для валидационных данных.
    """
    # Словарь, содержащий гиперпараметры модели SVC,
    # для каждой итерации trial генерируются разные.
    params = {
        # Меньшее значение C ведет к более широкой полосе, но большему
        # числу нарушений зазора.
        # При переобучении сократить С.
        "C": trial.suggest_categorical("C", [0.01, 1, 10, 100]),
        "kernel": trial.suggest_categorical(
            "kernel", ["linear", "poly", "rbf", "sigmoid"]
        ),
        # Если модель переобучается, тогда надо уменьшить значение
        # gamma, а если недообучается — то увеличить его.
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        "probability": True,
        "break_ties": trial.suggest_categorical("break_ties", [True, False]),
        "random_state": kwargs["random_state"],
    }

    if params["kernel"] == "poly":
        params["degree"] = trial.suggest_int("degree", 3, 5, 10)
    # coef0 управляет тем, насколько сильно полиномы высокой степени
    # влияют на модель в сравнении с полиномами низкой степени.
    if params["kernel"] in ["poly", "sigmoid"]:
        params["coef0"] = trial.suggest_float("coef0", 0.0, 5.0)

    # Т.к.в подбор параметров заложен элемент случайности, есть
    # вероятность, что вчерашние параметры могут оказаться лучше, т.к.
    # они уже подбирались на большем количестве многодневных итераций.
    # Но их надо проверить, замерив метрику на сегодняшних данных.

    # Вручную добавляем набор параметров для одной из итераций trial.
    # Используем trial.number для определения номера итерации.
    if trial.number == 0 and os.path.exists(kwargs["params_path"]):
        # Загружаем вчерашние параметры, если они существуют.
        with open(
            os.path.join(kwargs["params_path"]), "r", encoding='utf-8'
        ) as f:
            # Вручную задаём их в качестве параметров для текущей
            # итерации.
            params = json.load(f)

    # SVC имеет встроенную 5-fold кросс-валидацию при включённом
    # параметре probability.

    # Создание объекта модели SVC с гиперпараметрами, сгенерированными
    # для текущей итерации.
    model = SVC(**params)
    # Обучение.
    model.fit(*train_set)
    # Предсказание.
    y_pred = model.predict_proba(val_set[0])
    roc_auc_predict = roc_auc_score(
        y_true=val_set[1], y_score=y_pred, multi_class="ovr"
    )

    return roc_auc_predict


def study_with_optuna(
        train_set: Tuple[pd.DataFrame, pd.Series],
        val_set: Tuple[pd.DataFrame, pd.Series],
        **kwargs
) -> Study:
    """
    Обучает при помощи библиотеки Optuna для оптимизации
    гиперпараметров.

    Параметры:
        @param train_set: кортеж из
            датасета признаков для обучения модели и
            целевых значений соответствующих датасету признаков для
            обучения.
        @param val_set: кортеж из
            датасета признаков для валидации модели и
            целевых значений соответствующих датасету признаков для
            валидации.

    Ключи-аргументы:
        @kwarg study_path (str): Путь для сохранения результата Study.
        @kwarg params_path (str): Путь для сохранения оптимальных
            гиперпараметров в JSON формате.
        @kwarg value_path (str): Путь для сохранения лучшего значения
            метрики в JSON формате.

    Возвращает:
        @return: optuna.study.Study.
    """
    study_svc = optuna.create_study(
        direction="maximize", study_name="SVC_Optuna"
    )

    # Обёртка для вызова функции objective_svc с передачей аргументов
    # через объект trial.
    func = lambda trial: objective_svc(trial, train_set, val_set, **kwargs)

    study_svc.optimize(func, n_trials=10, show_progress_bar=True)

    # Сохранение результатов (study, best_params, best_value)
    joblib.dump(study_svc, os.path.join(kwargs["study_path"]))
    with open(os.path.join(kwargs["params_path"]), "w", encoding='utf-8') as f:
        json.dump(study_svc.best_params, f)
    with open(os.path.join(kwargs["value_path"]), "w", encoding='utf-8') as f:
        json.dump({"best_value": study_svc.best_value}, f)

    return study_svc


def train_on_best_params(
        best_params: dict, train_set: Tuple[pd.DataFrame, pd.Series], **kwargs
) -> SVC:
    """
    Обучает модель SVC с использованием оптимальных гиперпараметров.

    Параметры:
        @param best_params: Словарь с оптимальными гиперпараметрами для
            модели SVC.
        @param train_set: Кортеж из
            датасета признаков для обучения модели и
            целевых значений соответствующих датасету признаков для
            обучения.

    Ключи-аргументы:
        @kwarg model_path (str): Путь для сохранения обученной модели.
        @kwarg random_state (int): Число, фиксирующее начальное
            состояние для воспроизводимости результатов.

    Возвращает:
        @return: SVC: Обученная модель SVC.
    """
    svc = SVC(
        **best_params, random_state=kwargs["random_state"], probability=True
    )
    svc.fit(*train_set)
    # save model
    joblib.dump(svc, os.path.join(kwargs["model_path"]))

    return svc
