"""
Возможная доработка, при которой Оптюна будет подбирать параметры и для
SVC и для tf-idf.
"""
import optuna
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

def objective(trial):
    # Определение пространства поиска для параметров TfidfVectorizer
    vectorizer_params = {
        "max_df": trial.suggest_float("max_df", 0.7, 1.0),
        "max_features": trial.suggest_int("max_features", 1000, 10000),
    }

    # Определение пространства поиска для параметров модели SVC
    svc_params = {
        "C": trial.suggest_loguniform("C", 0.1, 10.0),
        "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"]),
    }

    # Создание экземпляра TfidfVectorizer с определенными параметрами
    vectorizer = TfidfVectorizer(**vectorizer_params)

    # Преобразование текстовых данных в матрицу TF-IDF
    X = vectorizer.fit_transform(texts)

    # Создание экземпляра модели SVC с определенными параметрами
    model = SVC(**svc_params)

    # Выполнение кросс-валидации с заданной моделью и метрикой
    scores = cross_val_score(model, X, labels, cv=5, scoring="accuracy")

    # Возвращение среднего значения точности как метрики оптимизации
    return scores.mean()

if __name__ == "__main__":
    # Создание экземпляра Study и запуск оптимизации
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    # Получение лучших параметров для TfidfVectorizer
    best_vectorizer_params = {
        "max_df": study.best_params["max_df"],
        "max_features": study.best_params["max_features"],
    }

    # Получение лучших параметров для модели SVC
    best_svc_params = {
        "C": study.best_params["C"],
        "kernel": study.best_params["kernel"],
    }

    # Вывод лучших параметров для TfidfVectorizer
    print("Лучшие параметры для TfidfVectorizer:")
    for param, value in best_vectorizer_params.items():
        print(f"{param}: {value}")

    # Вывод лучших параметров для модели SVC
    print("Лучшие параметры для модели SVC:")
    for param, value in best_svc_params.items():
        print(f"{param}: {value}")

    # Далее можно использовать эти лучшие параметры
    # для обучения модели с наилучшими найденными значениями.

# Если все признаки

# Определение пространства поиска для параметров TfidfVectorizer
vectorizer_params = {
    "max_df": trial.suggest_float("vectorizer_max_df", 0.7, 1.0),
    "max_features": trial.suggest_int("vectorizer_max_features", 1000, 10000),
    "input": trial.suggest_categorical("vectorizer_input", ["filename", "file", "content"]),
    "encoding": trial.suggest_categorical("vectorizer_encoding", ["utf-8", "latin1"]),
    "decode_error": trial.suggest_categorical("vectorizer_decode_error", ["strict", "ignore", "replace"]),
    "strip_accents": trial.suggest_categorical("vectorizer_strip_accents", [None, "ascii", "unicode"]),
    "lowercase": trial.suggest_categorical("vectorizer_lowercase", [True, False]),
    "preprocessor": trial.suggest_categorical("vectorizer_preprocessor", [None]),
    "tokenizer": trial.suggest_categorical("vectorizer_tokenizer", [None]),
    "analyzer": trial.suggest_categorical("vectorizer_analyzer", ["word", "char", "char_wb"]),
    "stop_words": trial.suggest_categorical("vectorizer_stop_words", [None, "english"]),
    "token_pattern": trial.suggest_categorical("vectorizer_token_pattern", [r"(?u)\\b\\w\\w+\\b", r"\b\w+\b"]),
    "ngram_range": trial.suggest_categorical("vectorizer_ngram_range", [(1, 1), (1, 2), (2, 2)]),
    "binary": trial.suggest_categorical("vectorizer_binary", [True, False]),
    "dtype": trial.suggest_categorical("vectorizer_dtype", ["float64", "int64"]),
    "norm": trial.suggest_categorical("vectorizer_norm", ["l1", "l2"]),
    "use_idf": trial.suggest_categorical("vectorizer_use_idf", [True, False]),
    "smooth_idf": trial.suggest_categorical("vectorizer_smooth_idf", [True, False]),
    "sublinear_tf": trial.suggest_categorical("vectorizer_sublinear_tf", [True, False]),
}

# Определение пространства поиска для параметров модели SVC
svc_params = {
    "C": trial.suggest_loguniform("svc_C", 0.1, 10.0),
    "kernel": trial.suggest_categorical("svc_kernel", ["linear", "rbf"]),
    "degree": trial.suggest_int("svc_degree", 1, 5),
    "gamma": trial.suggest_categorical("svc_gamma", ["scale", "auto"] + [trial.suggest_float("svc_gamma_custom", 0.001, 1.0)]),
    "coef0": trial.suggest_float("svc_coef0", -1.0, 1.0),
    "shrinking": trial.suggest_categorical("svc_shrinking", [True, False]),
    "probability": trial.suggest_categorical("svc_probability", [True, False]),
    "tol": trial.suggest_loguniform("svc_tol", 1e-5, 1e-1),
    "cache_size": trial.suggest_int("svc_cache_size", 100, 1000),
    "class_weight": trial.suggest_categorical("svc_class_weight", [None, "balanced"]),
    "verbose": trial.suggest_categorical("svc_verbose", [True, False]),
    "max_iter": trial.suggest_int("svc_max_iter", 100, 1000),
}
