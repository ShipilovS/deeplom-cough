import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.callbacks import ModelCheckpoint, EarlyStopping
import soundfile as sf

LABELS = ['VGMU_Coswata_Cutt', 'VGMU_Noise', 'VGMU_Voises_Cutt']
# Параметры
data_dir = '/home/sshipilov/Загрузки/Telegram Desktop/data_deeplom'
sample_rate = 44100
n_mfcc = 13
num_classes = 3
noise_factor = 0.05  # Интенсивность шума

# принудително юзать
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean, audio  # Вернуть и признаки и аудио

def add_noise(audio, noise_factor):
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    augmented_audio = augmented_audio.astype(type(audio[0]))
    return augmented_audio

# Сбор данных
X = []
y = []

for label in LABELS:
    folder_path = os.path.join(data_dir, label)
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            features, audio = extract_features(file_path)
            class_index = LABELS.index(label)

            # Оригинальный звук
            X.append(features)
            y.append(class_index)

            # Звук с шумом
            noisy_audio = add_noise(audio, noise_factor)
            noisy_features = librosa.feature.mfcc(y=noisy_audio, sr=sample_rate, n_mfcc=n_mfcc)
            noisy_features_mean = np.mean(noisy_features.T, axis=0)
            X.append(noisy_features_mean)
            y.append(class_index)  # Тот же класс, просто шумный звук


X = np.array(X)
y = np.array(y)

# encoder = OneHotEncoder(sparse_output=False) # для версии python-3.10
encoder = OneHotEncoder(sparse=False)# для версии python-3.7
y = encoder.fit_transform(y.reshape(-1, 1))
# Убедитесь, что входные данные имеют правильный размер
# X = X.reshape(X.shape[0], 1, X.shape[1], 1)


X = X.reshape(X.shape[0], 1, n_mfcc, 1)  # (n_samples, 1, 13, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

model = models.Sequential()

# Первый сверточный слой
model.add(layers.Conv2D(32, (1, 3), activation='relu',
                         input_shape=(1, n_mfcc, 1), kernel_regularizer=regularizers.l2(0.02)))
model.add(layers.BatchNormalization())  # Добавляем BatchNormalization
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling2D((1, 1)))

# Второй сверточный слой
model.add(layers.Conv2D(64, (1, 3), activation='relu', kernel_regularizer=regularizers.l2(0.02)))  # Добавляем регуляризацию
model.add(layers.BatchNormalization())  # Добавляем BatchNormalization
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling2D((1, 1)))

# Третий сверточный слой
model.add(layers.Conv2D(128, (1, 3), activation='relu', kernel_regularizer=regularizers.l2(0.02)))  # Добавляем регуляризацию
model.add(layers.BatchNormalization())  # Добавляем BatchNormalization
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling2D((1, 1)))

# Плоский слой
model.add(layers.Flatten())

# Полносвязный слой
model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.02)))  # Добавляем регуляризацию
model.add(layers.BatchNormalization())  # Добавляем BatchNormalization
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('cough_detection_model-37.h5',
                             monitor='val_accuracy',
                             save_best_only=True,
                             mode='max',
                             verbose=1)

early_stopping = EarlyStopping(monitor='val_accuracy', patience=10,
                               restore_best_weights=True)  # Добавляем EarlyStopping

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2,
          callbacks=[checkpoint, early_stopping])  # Добавляем EarlyStopping в callbacks

model.save('cough_detection_model-37_13_03.h5')
print("Модель сохранена в 'cough_detection_model.h5'")