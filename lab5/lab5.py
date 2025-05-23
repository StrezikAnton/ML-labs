#lab5
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import activations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler as Scaler
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

X = pd.read_csv("dataIn.txt", sep=' ', header=None)
y = pd.read_csv("dataOut.txt", sep=' ', header=None)

for i in range(0, len(X)) :
    if not(y[0][i] ^ y[1][i]) :
        X.drop(X.index[i])
        y.drop(y.index[i])

y.pop(1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Нормализация данных

scaler = Scaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


model = keras.Sequential([

    keras.layers.Dense(12, activation='sigmoid', input_shape=(12,)),  # входной слой
    
    keras.layers.Dense(6, activation='log_sigmoid'),
    
    keras.layers.Dense(1, activation='sigmoid')                    # Выходной слой (бинарная классификация)

])


# Компиляция модели

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Обучение модели

history = model.fit(X_train, y_train, epochs=50, batch_size=5, validation_data=(X_test, y_test))

# Оцениваем качество на тестовой выборке

y_pred = (model.predict(X_test) > 0.5).astype("int32")

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# График изменения функции ошибки

plt.figure("Loss");

plt.plot(history.history['loss'], label='Training Loss')

plt.plot(history.history['val_loss'], label='Validation Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()


plt.figure("Accuracy");

plt.plot(history.history['accuracy'], label='Training Accuracy')

plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()


plt.figure("predictions");

names_test, counts_test = np.unique_counts(y_test)
names_pred, counts_pred = np.unique_counts(y_pred)

print(names_test, names_pred)

names = ["1_test", "1_pred", "0_test", "0_pred"]

values = [counts_test[1], counts_pred[1], counts_test[0], counts_pred[0]]

plt.bar(names, values)

plt.show()
