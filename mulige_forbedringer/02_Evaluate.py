"""
Просто ещё один файл сгенерированный из ноутбука чтобы можно было брать
из него методы и другие куски кода.
"""
#!/usr/bin/env python
# coding: utf-8

# In[142]:


import pandas as pd
import joblib
import pymorphy2
import yaml
import json
import re
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer


# In[151]:


training


# In[154]:


evaluate


# # Import

# Загружаем новые, продовские данные

# In[155]:


data_eval_origin = pd.read_csv(evaluate['data_origin_path'], index_col=0, keep_default_na=False)
data_eval_origin


# # Preprocessing

# In[184]:


def get_data_text(data: pd.DataFrame, sum_columns: list) -> pd.DataFrame:
    """
    Объединяет текст из нескольких колонок в одну колонку, убирая небуквенные символы
    :param data: исходный датафрейм содержащий текстовые колонки
    :param sum_columns: список колонок для объединения
    :return: датафрейм содержащий одну колонку: 'text', которая содержит объединённый
        текст из колонок sum_columns
    """
    # Создаем пустой DataFrame с колонкой "text"
    data_text = pd.DataFrame(columns=['text'])
    # Добавляем столько же пустых строк, сколько строк в data
    for _ in range(data.shape[0]):
        data_text = data_text.append({'text': ''}, ignore_index=True)
    # Суммируем колонки в одну
    for i in range(len(sum_columns)):
        data_text['text'] += data[sum_columns[i]].astype(str) + ' '
    # Заменяем небуквенные символы на пробелы, а затем множественные пробелы на одинарные
    # и убираем пробелы в начале и в конце строки
    data_text['text'] = data_text['text'].apply(
        lambda x: re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Zа-яА-ЯЁё]', ' ', x)).strip())
    return data_text


def stem(lst: list, unnecessary_words: list) -> None:
    """
    Преобразует значения элементов листа, удаляя ненужные слова и
    приводя оставшиеся слова к нормальной форме
    :param lst: лист для изменения, элементы представляют из себя текстовые предложения
    :unnecessary_words: лист с ненужными словами
    :return: None, т.к. изменяется изначально поданный на функцию лист
    """
    morph = pymorphy2.MorphAnalyzer()

    for i in range(len(lst)):
        for word in unnecessary_words:
            lst[i] = re.sub(rf"\b{word}\b", "", lst[i], flags=re.IGNORECASE)
        lst[i] = " ".join(
            [morph.parse(word)[0].normal_form for word in lst[i].split()]
        )


def tf_idf(list_text: list, vectorizer_params: dict, verbosity: bool=False, csv_path: str=None) -> pd.DataFrame:
    """
    Из листа с текстовыми элементами формирует датафрейм с числовыми признаками
    по алгоритму TF-IDF
    :param list_text: лист с текстовыми признаками
    :param vectorizer_params: параметры для алгоритма TF-IDF
    :param verbosity: нужно ли выводить значения на экран
    :param csv_path: если не None, то путь для сохранения итогового датасета
    :return: датафрейм, названиями признаков которого являются слова из текстовых
        элементов изначального листа, а объектами -- числовые значения, рассчитанные
        по TF-IDF
    """
    vectorizer = TfidfVectorizer(**vectorizer_params)
    df_numbers = vectorizer.fit_transform(list_text).toarray()
    df_words = vectorizer.get_feature_names_out()
    numbers_words = pd.DataFrame(df_numbers, columns=df_words)

    if verbosity:
        print(f"df_numbers[:4] = \n{df_numbers[:4]}\n")
        print(f"df_words[:4] = \n{df_words[:4]}\n")
        print(f"numbers_words.head() = \n{numbers_words.head()}\n")
    
    if csv_path:
        numbers_words.to_csv(csv_path)
    
    return numbers_words


def get_data_num(data: pd.DataFrame) -> pd.DataFrame:
    """
    Создаёт датасет с числовыми признаками методом TF-IDF
    :param data: исходный датасет с текстовыми признаками
    :return: датасет с числовыми признаками, созданными методом TF-IDF
    """
    assert isinstance(data, pd.DataFrame), "Проблема с типом данных"
    # Создаём новый датасет с колонкой text
    data_text = get_data_text(data, preproc["sum_columns"])
    # Формируем питон лист из значений колонки text
    list_text = list(data_text["text"].values)
    # Чтобы в дальнейшем сократить количество признаков и упростить модель,
    # переведём слова в нормальную форму и уберём ненужные
    stem(lst=list_text,
        unnecessary_words=preproc["unnecessary_words"])
    # Применяем TF-IDF для создания датасета с числовыми признаками
    data_num = tf_idf(list_text=list_text,
                      vectorizer_params=preproc["vectorizer_params"],
                      csv_path=evaluate["data_num_path"])
    return data_num


def del_new_add_old_cols(data: pd.DataFrame, old_columns: list) -> None:
    """
    Удаляет из датасета те признаки, которых нет в листе, и
    добавляет с нулевыми значениями те признаки,
    которые есть в листе, но которых нет в датасете
    :param data: исходный датасет
    :param old_columns: список признаков
    :return: основная задача: изменяет исходный датасет, также
        возвращает число появившихся признаков и число пропавших признаков
    """
    # Удаляем признаки, которых нет в списке old_columns
    columns_to_remove = [col for col in data.columns if col not in old_columns]
    data.drop(columns_to_remove, axis=1, inplace=True)
    
    # Добавляем признаки из old_columns, которых нет в датасете
    columns_to_add = [col for col in old_columns if col not in data.columns]
    for col in columns_to_add:
        data[col] = 0.0  # Добавляем признак с нулевыми значениями
    return {
        'new_columns': len(columns_to_remove),
        'missed_columns': len(columns_to_add)
    }


def check_columns_evaluate(data: pd.DataFrame, train_proc_path: str) -> pd.DataFrame:
    """
    Проверка на наличие признаков из train и упорядочивание признаков согласно train
    :param data: датасет num_df
    :param train_proc_path: путь до датасета train
    :return: датасет с упорядоченными признаками
    """
    # Загружаем только строку с признаками обучающего датасета
    df_0_row = pd.read_csv(train_proc_path, nrows=0, index_col=0)
    # Переводим в список
    column_sequence = df_0_row.columns.tolist()

    if set(column_sequence) != set(data.columns):
        # Если признаки отличаются, то
        # вызываем метод, который
        # удалит новые признаки, которых нет в train, и
        # добавит в eval с нулевыми значениями старые признаки из train, которых нет в eval
        len_cols = del_new_add_old_cols(data, column_sequence)
    else:
        len_cols = {}
    # Упорядочиваем
    return data[column_sequence], len_cols


def pipeline_preprocess(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Пайплайн по предобработке данных
    :param data: оригинальный датасет
    :return: датасет готовый к предсказанию
    """
    # Создание датасета с числовыми признаками методом TF-IDF
    data_num = get_data_num(data)
    # проверка датасета на совпадение с признаками из train
    data_num, len_cols = check_columns_evaluate(data_num, kwargs["train_proc_path"])
    data_num.to_csv(kwargs["data_num_checked_cols_path"])

    return data_num, len_cols


# In[185]:


data_eval_num, len_cols = pipeline_preprocess(data=data_eval_origin,
                                               **preproc, **evaluate)



# # Evaluate

# In[163]:


model = joblib.load(training['model_path'])


# In[164]:


target_column_pred = preproc['target_column_pred']
data_eval_num[target_column_pred] = model.predict(data_eval_num)


# In[168]:


def get_df_predict_origin(df_predict: pd.DataFrame,
                          df_origin: pd.DataFrame,
                          drop_columns: list,
                          rename_columns: dict) -> pd.DataFrame:
    """
    Соединяет в один датафрейм предсказанные категории и изначальные
    :param df_predict: датафрейм с признаками и предсказанными значениями целевой
        переменной -- категории
    :param df_origin: датафрейм с признаками и изначальными значениями категории
    :param drop_columns: колонки для удаления
    :param rename_columns: колонки для переименования
    :return: датафрейм с признаками и предсказанными и изначальными значениями категории
    """
    df_origin = df_origin.reset_index(drop=True)
    # Соединяем только одну колонку cat_pred со всем датасетом df_origin
    df = pd.concat([df_predict[target_column_pred],
                    df_origin],
                   axis=1)
    # Удаляем ненужные и переименовываем столбцы
    df.drop(drop_columns, axis=1, inplace=True, errors='ignore')
    df.rename(rename_columns, axis=1, inplace=True, errors='ignore')
    
    return df


# In[169]:


data_eval_predict_origin = get_df_predict_origin(data_eval_num,
                                                 data_eval_origin,
                                                 preproc['drop_columns'],
                                                 preproc['rename_columns'])
data_eval_predict_origin


# In[ ]:





# Как видим, из 10ти значений:  
# - 4 -- точно верно предсказаны  
# - 4 -- пойдёт, но лучше бы подошла другая категория  
# - 2 -- точно не верно предсказаны  

# Чтобы оценить точность, выберем рандомно 100 строк и проверим вручную.
# Что будет грубой оценкой, но другой возможности оценить нет, поскольку у нас есть только предсказанные значения целевой переменной  

# In[189]:


random_sample = data_eval_predict_origin.sample(n=100)


# - 55 -- точно верно предсказаны (присваиваем 1 балл)  
# - 22 -- пойдёт, но лучше бы подошла другая категория (присваиваем 0.5 балла)  
# - 23 -- точно не верно предсказаны (присваиваем 0 баллов)  
# итого 66  

# Примерная точность 66%  
# 
# Такое значение связано с тем, что в evaluate сете много новых признаков и много недостающих,
# что значит много незнакомых для модели слов
# 

# In[187]:


len_cols


# 
# Учитывая, что на обучающем датасете точность была 96%,
# можно сделать вывод, что в дальнейшем расширяя обучающий датасет,
# мы добьёмся большей точности и на evaluate датасете
