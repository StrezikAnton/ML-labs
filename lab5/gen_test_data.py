# Генерация тестовых данных
import numpy as np
np.random.seed(42)
X = np.random.randint(0, 2, size=(100, 12))  # 100 примеров, 12 бинарных признаков
Y = np.random.randint(0, 2, size=(100, 2))  # 100 примеров, 2 класса (one-hot encoding)
np.savetxt('dataIn.txt', X, fmt='%d')
np.savetxt('dataOut.txt', Y, fmt='%d')
