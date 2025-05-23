# Импорт необходимых библиотек
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Загрузка предобработанных данных
train = pd.read_csv("preprocessed.csv")
test = pd.read_csv("preprocessed_test.csv")

# Разделение данных на признаки (X) и целевую переменную (y)
x_train = train.drop(['Transported_int'], axis='columns')  # Удаляем целевую переменную из признаков
y_train = train['Transported_int']  # Выделяем целевую переменную для обучения

x_test = test.drop(['Transported_int'], axis='columns')  # Аналогично для тестовых данных
y_test = test['Transported_int']

# Импорт и инициализация модели дерева решений
from sklearn.tree import DecisionTreeClassifier

# Создание модели с фиксированным random_state и ограничением глубины дерева
tree = DecisionTreeClassifier(random_state=42, max_depth=7)
tree.fit(x_train, y_train)  # Обучение модели на тренировочных данных

# Получение предсказаний модели
y_pred_train = tree.predict(x_train)  # Предсказания на тренировочных данных
y_pred_test = tree.predict(x_test)    # Предсказания на тестовых данных

# Вывод метрик для тестовых данных
print("test report")
report = classification_report(y_test, y_pred_test)
print(report)
print(confusion_matrix(y_test, y_pred_test))
print(f'Accuracy: {accuracy_score(y_test, y_pred_test):.2f}',"\n\n\n")

# Вывод метрик для тренировочных данных
print("train report")
reportTrain = classification_report(y_train, y_pred_train)
print(reportTrain)
print(confusion_matrix(y_train, y_pred_train))
print(f'Accuracy: {accuracy_score(y_train, y_pred_train):.2f}',"\n\n\n")

#Визуализация дерева решений (закомментировано)
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 8))
plot_tree(tree, filled=True)
plt.show()
