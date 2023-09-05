"""
Модуль: Преобразования данных.
"""

# Стандартные библиотеки
import json
import re

# Сторонние библиотеки
import pandas as pd
import pymorphy2
from sklearn.feature_extraction.text import TfidfVectorizer


def sum_text(df: pd.DataFrame, sum_columns: list) -> None:
    """
    Объединяет текст из нескольких колонок в одну колонку, убирая
    небуквенные символы.

    Параметры:
        @param df: Исходный датафрейм содержащий текстовые колонки.
        @param sum_columns: Список колонок для объединения.

    Возвращает:
        @return: None, функция изменяет исходный датафрейм добавляя новую
            колонку 'text', содержащую объединённый текст из заданных
            колонок.
    """
    df["text"] = ""
    # Суммируем колонки в одну.
    for column_name in sum_columns:
        df["text"] += df[column_name].astype(str) + " "
    # Заменяем небуквенные символы на пробелы, а затем множественные
    # пробелы на одинарные
    # и убираем пробелы в начале и в конце строки.
    df["text"] = df["text"].apply(
        lambda x: re.sub(
            r"\s+", " ", re.sub(r"[^a-zA-Zа-яА-ЯЁё]", " ", x)
        ).strip()
    )


def text_count(df: pd.DataFrame, col_name: str) -> None:
    """
    Считает количество слов в заданной исходной колонке и записывает его
    в новую колонку.

    Параметры:
        @param df: Исходный датафрейм содержащий текстовую колонку.
        @param col_name: Название исходной колонки для подсчёта слов.
    Возвращает:
        @return: None, функция изменяет исходный датафрейм добавляя
            новую колонку 'text_count', содержащую количество слов для
            каждого значения в заданной колонке.
    """
    df["text_count"] = df[col_name].apply(lambda x: len(x.split()))


def stem(lst: list, unnecessary_words: list) -> None:
    """
    Преобразует значения элементов листа, удаляя ненужные слова и
    приводя оставшиеся слова к нормальной форме.

    Параметры:
        @param lst: Лист для изменения, элементы представляют собой
            текстовые предложения.
        @param unnecessary_words: Лист с ненужными словами.

    Возвращает:
        @return: None, т.к. изменяется изначально поданный на функцию
            лист.
    """
    morph = pymorphy2.MorphAnalyzer()

    for i, _ in enumerate(lst):
        for word in unnecessary_words:
            lst[i] = re.sub(rf"\b{word}\b", "", lst[i], flags=re.IGNORECASE)
        lst[i] = " ".join(
            [morph.parse(word)[0].normal_form for word in lst[i].split()]
        )


def tf_idf(
    list_text: list,
    vectorizer_params: dict,
) -> pd.DataFrame:
    """
    Из листа с текстовыми элементами формирует датафрейм с числовыми
    признаками по алгоритму TF-IDF.

    Параметры:
        @param list_text: Лист с текстовыми признаками.
        @param vectorizer_params: Параметры для алгоритма TF-IDF.

    Возвращает:
        @return: Датафрейм, названиями признаков которого являются слова
            из текстовых элементов изначального листа, а объектами --
            числовые значения, рассчитанные по TF-IDF.
    """
    # Создание объекта TfidfVectorizer с переданными параметрами
    vectorizer = TfidfVectorizer(**vectorizer_params)
    # Преобразование списка текстовых данных list_text в матрицу
    # числовых значений (Tfidf-признаки)
    df_numbers = vectorizer.fit_transform(list_text).toarray()
    # Получение списка слов (фичей) из TfidfVectorizer
    df_words = vectorizer.get_feature_names_out()
    # Создание DataFrame с числовыми признаками (Tfidf-значениями) и их
    # именами (словами-фичами)
    numbers_words = pd.DataFrame(df_numbers, columns=df_words)

    return numbers_words


def del_new_add_old_cols(
    df: pd.DataFrame, old_columns: list, **kwargs
) -> None:
    """
    Удаляет из датасета те признаки, которых нет в листе;
    и добавляет с нулевыми значениями те признаки, которые есть в листе,
    но которых нет в датасете.

    Параметры:
        @param df: Исходный датасет.
        @param old_columns: Список признаков.

    Возвращает:
        @return: None. Изменяет исходный датасет.
    """
    # Удаляем признаки, которых нет в списке old_columns.
    columns_to_remove = [col for col in df.columns if col not in old_columns]
    df.drop(columns_to_remove, axis=1, inplace=True)

    # Добавляем признаки из old_columns, которых нет в датасете.
    columns_to_add = [col for col in old_columns if col not in df.columns]
    for col in columns_to_add:
        df[col] = 0.0  # Добавляем признак с нулевыми значениями.

    len_cols = {
        "new_columns": len(columns_to_remove),
        "missed_columns": len(columns_to_add),
    }
    # Сохраняем len_cols.
    with open(kwargs["len_cols_path"], "w", encoding='utf-8') as f:
        json.dump(len_cols, f)


def check_columns_evaluate(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Проверяет на наличие признаков из train и упорядочивает признаки
    согласно train.

    Параметры:
        @param df: Датафрейм с числовыми признаками.

    Ключи-аргументы:
        @kwarg train_num_df_path: Путь до train датасета.

    Возвращает:
        @return: Датафрейм с упорядоченными признаками.
    """
    # Загружаем только строку с названиями признаков обучающего датасета
    # (Если датасет огромный, то загрузить одну строку быстрее чем весь)
    df_0_row = pd.read_csv(kwargs["train_num_df_path"], nrows=0, index_col=0)
    # Переводим в список
    column_sequence = df_0_row.columns.tolist()

    if set(column_sequence) != set(df.columns):
        # Если признаки отличаются, то
        # вызываем метод, который
        # удалит новые признаки, которых нет в train, и
        # добавит в eval с нулевыми значениями старые признаки из train,
        # которых нет в eval.
        del_new_add_old_cols(df, column_sequence, **kwargs)
    # Упорядочиваем
    return df[column_sequence]
