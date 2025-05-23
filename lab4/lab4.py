# Импорт необходимых библиотек
import pandas as pd
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt

# Загрузка предобработанных данных
train = pd.read_csv("preprocessed.csv")
test = pd.read_csv("preprocessed_test.csv")

# Разделение данных на признаки (X) и целевую переменную (y)
x_train = train.drop(['Transported_int'], axis='columns')  # Удаление целевой переменной из признаков
y_train = train['Transported_int']                        # Целевая переменная для обучения

x_test = test.drop(['Transported_int'], axis='columns')   # Аналогично для тестовых данных
y_test = test['Transported_int']

# Создание и обучение модели Random Forest
rf = RandomForestClassifier(
    random_state=42,   # Фиксируем seed для воспроизводимости
    max_depth=7        # Ограничиваем глубину деревьев для борьбы с переобучением
)
rf.fit(x_train, y_train)  # Обучение модели на тренировочных данных

# Получение предсказаний
rf_pred = rf.predict(x_test)        # Предсказания на тестовых данных
rf_pred_train = rf.predict(x_train) # Предсказания на тренировочных данных

# Создание и обучение модели Gradient Boosting
gb = GradientBoostingClassifier(
    random_state=42    # Фиксируем seed для воспроизводимости
)                      # Параметры по умолчанию: n_estimators=100, learning_rate=0.1
gb.fit(x_train, y_train)

# Получение предсказаний
gb_pred = gb.predict(x_test)        # Предсказания на тестовых данных
gb_pred_train = gb.predict(x_train) # Предсказания на тренировочных данных

# Вычисление метрик для тестовых данных
accuracy = [
    metrics.accuracy_score(y_test, rf_pred)*100,
    metrics.accuracy_score(y_test, gb_pred)*100
]
f1_score = [
    metrics.f1_score(y_test, rf_pred)*100,
    metrics.f1_score(y_test, gb_pred)*100
]
recall = [
    metrics.recall_score(y_test, rf_pred)*100,
    metrics.recall_score(y_test, gb_pred)*100
]
precision = [
    metrics.precision_score(y_test, rf_pred)*100,
    metrics.precision_score(y_test, gb_pred)*100
]

# Аналогичные метрики для тренировочных данных
accuracy_train = [
    metrics.accuracy_score(y_train, rf_pred_train)*100,
    metrics.accuracy_score(y_train, gb_pred_train)*100
]
f1_score_train = [
    metrics.f1_score(y_train, rf_pred_train)*100,
    metrics.f1_score(y_train, gb_pred_train)*100
]
recall_train = [
    metrics.recall_score(y_train, rf_pred_train)*100,
    metrics.recall_score(y_train, gb_pred_train)*100
]
precision_train = [
    metrics.precision_score(y_train, rf_pred_train)*100,
    metrics.precision_score(y_train, gb_pred_train)*100
]

# Подготовка данных для визуализации
models = [
    'RF_accuracy', 'GB_accuracy',
    'RF__f1_score', 'GB__f1_score',
    'RF_recall', 'GB_recall',
    'RF_precision', 'GB_precision'
]

meters = accuracy + f1_score + recall + precision          # Метрики теста
meters_train = accuracy_train + f1_score_train + recall_train + precision_train  # Метрики трейна

# Настройка графиков
plt.figure(num="test", figsize=(13, 4))
plt.bar(models, meters)
plt.ylim(75, 85)               # Ограничение оси Y для лучшей визуализации
plt.title("Сравнение моделей (Тестовые данные)")
plt.ylabel("%")

plt.figure(num="train", figsize=(13, 4))
plt.bar(models, meters_train)
plt.ylim(75, 90)               # Диапазон для тренировочных данных
plt.title("Сравнение моделей (Тренировочные данные)")
plt.ylabel("%")

plt.show()
