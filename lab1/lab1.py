import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler # для 5 задания
# 1)указываю путь к файлу
file_path = "/home/anton/train.csv"
data = pd.read_csv(file_path)
data.info()

# 2)вывожу данные из датасета не экран
data_size = data.shape # размер датасета
print(data_size)
print(data.head(data_size[0]))

# 3) пропущенные значения в датасете
missing_values = data.isnull().sum()
print("Пропущенные значения: ", missing_values)

# 4) Заполнение пропущенных значений при помощи моды, медианы и среднего значения
# заполним пропуске в возрасте при помощи медианы
data['Age'] = data['Age'].fillna(data['Age'].median())
# заполним пропуски в HomePlanet при помощи моды
data['HomePlanet'] = data['HomePlanet'].fillna(data['HomePlanet'].mode()[0])
# заполним все остальные значения
data['CryoSleep'] = data['CryoSleep'].fillna(data['CryoSleep'].mode()[0])
data['Cabin'] = data['Cabin'].fillna(data['Cabin'].mode()[0])
data['Destination'] = data['Destination'].fillna(data['Destination'].mode()[0])
data['VIP'] = data['VIP'].fillna(data['VIP'].mode()[0])
data['RoomService'] = data['RoomService'].fillna(data['RoomService'].mean())
data['FoodCourt'] = data['FoodCourt'].fillna(data['FoodCourt'].mean())
data['ShoppingMall'] = data['ShoppingMall'].fillna(data['ShoppingMall'].mean())
data['Spa'] = data['Spa'].fillna(data['Spa'].mean())
data['VRDeck'] = data['VRDeck'].fillna(data['VRDeck'].mean())
# отдельная ситуация с именами, которую надо решить (я решил заполнить распространенными именами)
default_names = ["Anton", "Mikita", "John", "Michael", "Alex"]
default_surnames = ["Smith", "Simpson", "Kuznecov", "Popov", "Socolov"]
data['Name'] = data['Name'].fillna(random.choice(default_names) + " " + random.choice(default_surnames)) # такой способ не очень хорош, но вообщем сойдет
print(data.isnull().sum())

# 5) нормализация данных (нормализацию проведем только в числовых столбцах)
num_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
# Применяем MinMaxScaler
scaler = MinMaxScaler()
data[num_columns] = scaler.fit_transform(data[num_columns])
print(data[num_columns].head())

# 6) Преобразование категориальных данных

binary_values = pd.get_dummies(data['CryoSleep'], prefix='CryoSleep')
data = pd.concat([data, binary_values], axis=1)
data.drop('CryoSleep', axis=1, inplace=True)
# Выводим созданные бинарные столбцы
print(data[['CryoSleep_False', 'CryoSleep_True']].head())

binary_values1 = pd.get_dummies(data['Transported'], prefix='Transported')
data = pd.concat([data, binary_values1], axis=1)
data.drop('Transported', axis=1, inplace=True)
print(data[['Transported_False', 'Transported_True']].head())

binary_values2 = pd.get_dummies(data['VIP'], prefix='VIP')
data = pd.concat([data, binary_values2], axis=1)
data.drop('VIP', axis=1, inplace=True)
print(data[['VIP_False', 'VIP_True']].head())


# binary_values = pd.get_dummies(data['CryoSleep'], prefix='CryoSleep')
# data = pd.concat([data, binary_values], axis=1)
# data.drop('CryoSleep', axis=1, inplace=True)
# print(data['CryoSleep'])

# Сохраняем датасет в CSV файл
# output_file_path = "/home/anton/processed_dataset.csv"
# data.to_csv(output_file_path, index=False)
# print(f"Итоговый датасет сохранён по адресу: {output_file_path}")
