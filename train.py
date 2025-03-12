import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import layers, models
from keras.utils import pad_sequences

LABELS = ['VGMU_Coswata_Cutt', 'VGMU_Noise', 'VGMU_Voises_Cutt']
# Параметры
# data_dir = '/home/sshipilov/Загрузки/Telegram Desktop/data_deeplom'
# data_dir = '/home/sshipilov/Загрузки/Telegram Desktop/VGMU_Noise' 
# data_dir = '/home/sshipilov/Загрузки/Telegram Desktop/VGMU_Voises_Cutt'
data_dir = '/home/sshipilov/python_projects/cough_real_time'
sample_rate = 44100
n_mfcc = 13
num_classes = 3

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# Сбор данных
X = []
y = []

for label in LABELS:
    folder_path = os.path.join(data_dir, label)
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            features = extract_features(file_path)
            X.append(features)
            y.append(LABELS.index(label)) 

X = np.array(X)
y = np.array(y)

encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1))

X = X.reshape(X.shape[0], 1, X.shape[1], 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

print(f"train")
# Создание модели
model = models.Sequential()
model.add(layers.Conv2D(32, (1, 3), activation='relu', input_shape=(1, 13, 1)))
model.add(layers.MaxPooling2D((1, 2)))
model.add(layers.Conv2D(64, (1, 3), activation='relu'))
model.add(layers.MaxPooling2D((1, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

model.save('cough_detection_model.h5')
print("Модель сохранена в 'cough_detection_model.h5'")