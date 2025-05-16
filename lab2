import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split

# Загрузка обработанного датасета
train_data = pd.read_csv("/home/anton/processed_dataset.csv")

# Преобразование категориальных признаков в числовые с помощью one-hot encoding
# Выбираем категориальные колонки, которые не были обработаны в первой лабе
categorical_columns = ['HomePlanet', 'Destination', 'Cabin', 'Name']
train_data = pd.get_dummies(train_data, columns=categorical_columns)

# Определение признаков и целевой переменной
# Целевая переменная - Transported_True (был транспортирован)
X = train_data.drop(['PassengerId', 'Transported_False', 'Transported_True'], axis=1)  # Удаляем ID и оба столбца Transported
Y = train_data['Transported_True']  # Используем один из бинарных столбцов как целевую переменную

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Обучение модели логистической регрессии
log_model = LogisticRegression(max_iter=1000)  # Увеличиваем количество итераций для сходимости
log_model.fit(X_train, Y_train)
Y_pred_log = log_model.predict(X_test)

# Оценка качества логистической регрессии
recall_logistic = recall_score(Y_test, Y_pred_log)
accuracy_log = accuracy_score(Y_test, Y_pred_log)
print(f'Accuracy (Logistic Regression): {accuracy_log:.2f}')
print(f"Recall (Logistic Regression): {recall_logistic:.2f}")

# Обучение модели линейной регрессии с последующей бинаризацией
lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)
Y_pred_lin = lin_model.predict(X_test)
Y_pred_lin = (Y_pred_lin >= 0.5).astype(int)  # Преобразование вероятностей в классы

# Оценка качества линейной регрессии
recall_lin = recall_score(Y_test, Y_pred_lin)
accuracy_lin = accuracy_score(Y_test, Y_pred_lin)
print(f'Accuracy (Linear Regression): {accuracy_lin:.2f}')
print(f"Recall (Linear Regression): {recall_lin:.2f}")

# Построение матриц ошибок
cm_log = confusion_matrix(Y_test, Y_pred_log)
cm_lin = confusion_matrix(Y_test, Y_pred_lin)

print('Матрица ошибок логистической регрессии:\n', cm_log)
print('Матрица ошибок линейной регрессии:\n', cm_lin)

# Сохранение метрик в CSV
metrics_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Linear Regression'],
    'Accuracy': [accuracy_log, accuracy_lin],
    'Recall': [recall_logistic, recall_lin]
})
metrics_df.to_csv('regression_metrics.csv', index=False)

# Сохранение матриц ошибок
cm_log_df = pd.DataFrame(cm_log,
                        index=['Не транспортирован', 'Транспортирован'],
                        columns=['Не транспортирован', 'Транспортирован'])
cm_log_df.to_csv('logistic_cm.csv')

cm_lin_df = pd.DataFrame(cm_lin,
                        index=['Не транспортирован', 'Транспортирован'],
                        columns=['Не транспортирован', 'Транспортирован'])
cm_lin_df.to_csv('linear_cm.csv')

# Модифицированная часть с визуализацией
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Логистическая регрессия
sns.heatmap(cm_log, annot=True, fmt="d", cmap="coolwarm",
           xticklabels=['Не транспортирован', 'Транспортирован'],
           yticklabels=['Не транспортирован', 'Транспортирован'],
           ax=axes[0])
axes[0].set_xlabel("Предсказанный класс")
axes[0].set_ylabel("Истинный класс")
axes[0].set_title("Логистическая регрессия")

# Линейная регрессия
sns.heatmap(cm_lin, annot=True, fmt="d", cmap="coolwarm",
           xticklabels=['Не транспортирован', 'Транспортирован'],
           yticklabels=['Не транспортирован', 'Транспортирован'],
           ax=axes[1])
axes[1].set_xlabel("Предсказанный класс")
axes[1].set_ylabel("Истинный класс")
axes[1].set_title("Линейная регрессия")

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')  # Сохранение графиков
plt.show()
