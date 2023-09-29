creatives_classification_production
==============================

## UI Demo
![alt text](demo/Gifius.gif?raw=true)
![alt text](demo/orig.gif?raw=true)

Классифицирует рекламные видео креативы по категориям на основе их 
текстового описания.
Для работы с проектом необходимо использовать данные для тренировки и 
предсказания в формате csv с текстовыми полями title, description, adomain,
bundle; тренировочный датасет также должен содержать текстовое поле с целевой
переменной cat.

## Запуск приложения

- Для запуска приложения внутри докера достаточно использовать команду

`docker compose up`  
- Доступ к бэкенду (если прописан 8000 порт)  
http://localhost:8000/docs  
- Доступ к фронтенду (если прописан 8501 порт)  
http://localhost:8501  

Разделы приложения
------------  
- `Description` - Описание текстовых полей.  
- `Training` - Тренировка модели.  
   - Запуск тренировки.  
   - Отображение метрик.  
   - Отображение лучших гиперпараметров и их важности.  
   - Отображение истории оптимизации.
- `Prediction from file` - Предсказание из файла.  
   - Запуск предсказания значений целевой переменной на рабочем датасете.  
   - Отображение результатов предсказания.
- `Prediction from input` - Предсказание по введённым вручную данным.  
   - Запуск предсказания значений целевой переменной на 
  одном объекте.  
   Значения признаков вводятся вручную в форму.
- `Analysis & Visualization` - Отображение графиков, метрик и значений.  
   Для предсказания из файла.  
   - Распределение кол-ва слов в объектах.  
   - Количество объектов в категории тренировочного датасета.  
   - Количество объектов в предсказанной категории рабочего 
датасета.  
   - Диаграмма рассеяния предсказанных значений категории рабочего 
датасета.  
   - Количество новых и недостающих слов в рабочем датасете по 
сравнению с обучающим.

Файловая структура проекта
------------
Для упрощения не указаны аналогичные файлы и папки, а также часто используемые.


    
    ├── backend                <- Бэкенд FastAPI часть проекта.
    │   ├── requirements.txt   <- Файл со списком использованных библиотек
    │   │                         с номерами их версий.
    │   └── src                <- Исходный код бэкенда.
    │       ├── data           <- Методы для работы с данными.
    │       ├── features       <- Методы для обработки и создания признаков.
    │       ├── pipelines      <- Пайплайны предобработки данных, тренировки 
    │       │                     модели и предсказания.
    │       └── train          <- Методы используемые при тренировке модели.  
    ├── config                 <- Yaml файл содержащий все параметры, 
    │                             такие как пути к датасетам, моделям, 
    │                             ендпойнты и пр.
    ├── data                   <- Директория с табличными и вспомогательными 
    │   │                         данными.
    │   ├── check              <- Рабочие данные для предсказания.
    │   ├── processed          <- Обработанные данные для использования
    │   │                         моделью.
    │   └── raw                <- Изначальные данные для тренировки модели.
    ├── frontend               <- Фронтенд Streamlit часть проекта.
    │   └── src                <- Исходный код фронтенда.
    │       ├── evaluate       <- Методы для предсказания.
    │       ├── train          <- Методы для тренировки.
    │       └── visualization  <- Методы для отображения графиков, 
    │                             метрик и значений.
    ├── models                 <- Тренированные модели и объекты обучения Study.
    ├── notebooks              <- Jupyter ноутбуки использованные для
    │                             предварительной исследовательской работы:
    │                             анализа и тренировки.
    └── reports                <- Сгенерированные метрики и значения.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
