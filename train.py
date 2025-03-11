import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Параметры
data_dir = '/home/sshipilov/Загрузки/VGMU_Coswata_Cutt'  # Путь к папке с данными
data_dir = '/home/sshipilov/Загрузки/VGMU_Noise'  # Путь к папке с данными
data_dir = '/home/sshipilov/Загрузки/VGMU_Voises_Cutt'  # Путь к папке с данными
sample_rate = 22050
n_mfcc = 13

# Функция для извлечения MFCC
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=sample_rate)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

# Сбор данных
X = []
y = []

for label in ['coughs', 'noises']:
    folder_path = os.path.join(data_dir, label)
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            features = extract_features(file_path)
            X.append(features)
            y.append(label)

# Преобразование в массивы NumPy
X = np.array(X)
y = np.array(y)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Тестирование модели
y_pred = model.predict(X_test)

# Оценка результатов
print(classification_report(y_test, y_pred))
