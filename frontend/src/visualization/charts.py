"""
Программа: Отрисовка графиков
Версия: 1.0
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.axes as axes
from umap import UMAP


def format_json_as_list(json_obj, indent=0):
    """
    Функция для форматирования JSON-данных в виде списка

    @param json_obj:
    @param indent:
    @return:
    """
    output = []
    for key, value in json_obj.items():
        if isinstance(value, dict):
            output.append(f"{' ' * indent}{key}:")
            output.extend(format_json_as_list(value, indent + 4))
        else:
            output.append(f"{' ' * indent}{key}: {value}")
    return output


def total_len_words(df_path: str) -> int:
    """
    Считает количество признаков в датасете по заданному пути.
    """
    # Загружаем только первую строку с именами признаков.
    features = pd.read_csv(df_path, nrows=0, index_col=0)
    features_list = features.columns.tolist()
    return len(features_list)


def words_count_bars(
        data_frame: pd.DataFrame, col: str
) -> matplotlib.figure.Figure:
    fig = plt.figure(figsize=(15, 7))

    plt.title('Распределение кол-ва слов в объектах', fontsize=14)
    plt.hist(x=col, data=data_frame, bins=40)

    plt.xlabel('Кол-во слов в объекте')
    plt.ylabel('Кол-во объектов')
    return fig


def plot_text(ax: axes.Axes) -> None:
    """
    Добавляет аннотации с процентными значениями к каждому столбцу на столбчатой диаграмме
    :param ax: область для построения графиков в системе координат
    :return: None
    """
    # Для каждого столбца p
    for p in ax.patches:
        # Получение значения процента в формате строки
        percentage = '{:.1f}%'.format(p.get_width())
        ax.annotate(
            # Текст аннотации
            percentage,
            # Координата xy
            (p.get_width(), p.get_y() + p.get_height()),
            # Центрирование текста
            ha='center',
            va='center',
            xytext=(20, 10),
            # Точка смещения относительно координаты
            textcoords='offset points',
            fontsize=14)


def barplot_category_percents(sr: pd.Series, title: str) -> None:
    """
    Строит барплот с количеством объектов в процентном соотношении
    по категориям.
    :param title: название барплота
    :param sr: пандас серия, для которой будет строиться барплот
    :return: Рисунок с отображением барплота
    """
    # Нормирование на размер датасета
    norm_target = (sr
                   .value_counts(normalize=True)
                   .mul(100)
                   .rename('percent')
                   .reset_index())

    fig = plt.figure(figsize=(15, 7))
    ax = sns.barplot(y='index', x='percent', data=norm_target, orient='h')

    plot_text(ax)

    plt.title(title, fontsize=20)
    plt.xlabel('Проценты', fontsize=14)
    plt.ylabel('Категория', fontsize=14)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return fig


def embeddings_diagram(X: pd.DataFrame, y_pred, title: str, random_state: int
                       ) -> None:
    """
    Принимает набор объект-признаков и предсказанные для него значения
    целевой переменной. Создаёт эмбеддинги сниженной размерности.
    Рисует диаграмму рассеяния.
    :param y_pred: предсказанные значения целевой переменной
    :param X: набор данных
    :param title: наименование диаграммы
    :param random_state: Число, фиксирующее состояние для воспроизводимости
        результатов.
    :return: Рисунок с диаграммой рассеяния.
    """
    um = UMAP(n_components=3, random_state=random_state, n_neighbors=15,
              min_dist=0.1)
    X_embedding = um.fit_transform(X)

    fig = plt.figure(figsize=(13, 8))
    plt.title(title, fontsize=14)
    sns.scatterplot(x=X_embedding[:, 0],
                    y=X_embedding[:, 1],
                    hue=y_pred,
                    s=100,
                    legend='full')
    return fig
