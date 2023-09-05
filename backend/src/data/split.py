"""
Модуль: Разделение данных.
"""

# Стандартные библиотеки.
from typing import Tuple

# Сторонние библиотеки.
import pandas as pd
from sklearn.model_selection import train_test_split


def train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    **kwargs
) -> Tuple[
    Tuple[pd.DataFrame, pd.Series],
    Tuple[pd.DataFrame, pd.Series],
    Tuple[pd.DataFrame, pd.Series],
]:
    """
    Разбивает данные на тренировочный, валидационный и тестовый наборы.

    Параметры:
        @param X: Датафрейм с объект-признаками.
        @param y: Серия содержащая целевую переменную.
        **kwargs: Словарь дополнительных параметров из конфигурационного
            файла. Используются в функции train_test_split().

    Ключи-аргументы:
        @kwarg test_size: Пропорция размера тестового сета.
        @kwarg val_size: Пропорция размера валидационного сета.
        @kwarg random_state: Числовое значение для обеспечения
            воспроизводимости результата.

    Возвращает:
        @return: Три тюпла:
            - Тренировочные данные и целевая переменная.
            - Валидационные данные и целевая переменная.
            - Тестовые данные и целевая переменная.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=kwargs["test_size"],
        random_state=kwargs["random_state"],
        shuffle=True,
        # Указываем stratify, чтобы захватить и в test и в train по
        # объектам из каждого класса, чтобы потом посчитать рок аук.
        stratify=y,
    )

    X_train_, X_val, y_train_, y_val = train_test_split(
        X_train,
        y_train,
        test_size=kwargs["val_size"],
        random_state=kwargs["random_state"],
        shuffle=True,
        stratify=y_train,
    )

    train_set = (X_train_, y_train_)
    val_set = (X_val, y_val)
    test_set = (X_test, y_test)

    return train_set, val_set, test_set
