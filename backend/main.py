"""
Модуль: Модель машинного обучения для классификации рекламных видео
креативов по категориям на основе их текстовых описаний.
"""

# Стандартные библиотеки.
import warnings

# Сторонние библиотеки.
import optuna
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

# Локальные модули.
from src.pipelines.pipeline_train import pipeline_train
from src.pipelines.pipeline_evaluate import pipeline_evaluate
from src.train.metrics import load_metrics

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

app = FastAPI()
CONFIG_PATH = "../config/params.yml"


class Creative(BaseModel):
    """
    Текстовые поля креатива.
    """

    # Заголовок рекламаного креатива, обычно это название рекламируемого
    # продукта.
    Title: str
    # Описание рекламного ролика, более подробная информация о ролике и
    # продукте.
    Description: str
    # Домен рекламодателя, зачастую содержит название продукта.
    Adomain: str
    # Наименование бандла, также может содержать название продукта.
    Bundle: str


@app.get("/hello")
def welcome():
    """
    Приветствует.

    Возвращает:
        @return: Приветственное сообщение.
    """
    return {"message": "Hello Data Scientist! Это модель машинного"
                       "обучения для классификации рекламных видео"
                       "креативов по категориям на основе их текстовых"
                       "описаний."}


@app.post("/train")
def training():
    """
    Обучает модель, логирует метрики.

    Возвращает:
        @return: Json с метриками.
    """
    pipeline_train(config_path=CONFIG_PATH)
    metrics = load_metrics(config_path=CONFIG_PATH)

    return {"metrics": metrics}


@app.post("/predict")
def prediction(file: UploadFile = File(...)):
    """
    Отдаёт модели на предсказание по данным из файла.

    Возвращает:
        @return: Json с предсказаниями.
    """
    result = pipeline_evaluate(config_path=CONFIG_PATH, df_path=file.file)
    assert isinstance(result, list), "Результат не соответствует типу list"
    # заглушка так как не выводим все предсказания, иначе зависнет
    return {"prediction": result[:5]}


@app.post("/predict_input")
def prediction_input(creative: Creative):
    """
    Отдаёт модели на предсказание по вручную введенным данным.

    Возвращает:
        @return: Json с предсказанием.
    """
    features = [
        [
            creative.Title,
            creative.Description,
            creative.Adomain,
            creative.Bundle,
        ]
    ]

    cols = [
        "Title",
        "Description",
        "Adomain",
        "Bundle",
    ]

    data = pd.DataFrame(features, columns=cols)
    prediction_result = pipeline_evaluate(
        config_path=CONFIG_PATH, data_frame=data
    )[0]

    return prediction_result


if __name__ == "__main__":
    # Запустите сервер, используя заданный хост и порт
    uvicorn.run(app, host="127.0.0.1", port=80)
