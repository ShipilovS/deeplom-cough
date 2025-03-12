import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import layers, models, regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import pad_sequences
import tensorflow as tf

# Обучение модели с аугментацией

LABELS = ['VGMU_Coswata_Cutt', 'VGMU_Noise', 'VGMU_Voises_Cutt']
# Параметры
data_dir = '/home/sshipilov/Загрузки/Telegram Desktop/data_deeplom'
# data_dir = '/home/sshipilov/Загрузки/Telegram Desktop/VGMU_Noise' 
# data_dir = '/home/sshipilov/Загрузки/Telegram Desktop/VGMU_Voises_Cutt'
# data_dir = '/home/sshipilov/python_projects/cough_real_time'
sample_rate = 44100
n_mfcc = 13
num_classes = 3

# принудително юзать
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

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

# encoder = OneHotEncoder(sparse_output=False) # для версии python-3.10
encoder = OneHotEncoder(sparse=False)# для версии python-3.7
y = encoder.fit_transform(y.reshape(-1, 1))
# Убедитесь, что входные данные имеют правильный размер
# X = X.reshape(X.shape[0], 1, X.shape[1], 1)


X = X.reshape(X.shape[0], 1, n_mfcc, 1)  # (n_samples, 1, 13, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Создание улучшенной модели
model = models.Sequential()

# Первый сверточный слой
model.add(layers.Conv2D(32, (1, 3), activation='relu', 
                         input_shape=(1, n_mfcc, 1), kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling2D((1, 1)))

# Второй сверточный слой
model.add(layers.Conv2D(64, (1, 3), activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling2D((1, 1)))

# Третий сверточный слой
model.add(layers.Conv2D(128, (1, 3), activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling2D((1, 1)))

# Плоский слой
model.add(layers.Flatten())

# Полносвязный слой
model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(0.5)) 
model.add(layers.Dense(num_classes, activation='softmax'))

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('cough_detection_model-37.h5', 
                             monitor='val_accuracy', 
                             save_best_only=True, 
                             mode='max', 
                             verbose=1)

model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.2, callbacks=[checkpoint])

model.save('cough_detection_model-37.h5')
print("Модель сохранена в 'cough_detection_model.h5'")

# 15/15 [==============================] - 0s 9ms/step - loss: 0.5010 - accuracy: 0.8324 - val_loss: 0.4787 - val_accuracy: 0.8253
# Epoch 996/1000
# 15/15 [==============================] - 0s 10ms/step - loss: 0.4658 - accuracy: 0.8226 - val_loss: 0.4777 - val_accuracy: 0.8428
# Epoch 997/1000
# 15/15 [==============================] - 0s 10ms/step - loss: 0.4521 - accuracy: 0.8423 - val_loss: 0.4794 - val_accuracy: 0.8297
# Epoch 998/1000
# 15/15 [==============================] - 0s 9ms/step - loss: 0.4552 - accuracy: 0.8379 - val_loss: 0.4841 - val_accuracy: 0.8166
# Epoch 999/1000
# 15/15 [==============================] - 0s 10ms/step - loss: 0.4629 - accuracy: 0.8532 - val_loss: 0.4852 - val_accuracy: 0.8166
# Epoch 1000/1000
# 15/15 [==============================] - 0s 10ms/step - loss: 0.4660 - accuracy: 0.8412 - val_loss: 0.4687 - val_accuracy: 0.8297
