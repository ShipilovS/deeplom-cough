import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import layers, models, regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Параметры
LABELS = ['VGMU_Coswata_Cutt', 'VGMU_Noise', 'VGMU_Voises_Cutt']
data_dir = '/home/sshipilov/Загрузки/Telegram Desktop/data_deeplom'
data_dir = '/home/sshipilov/python_projects/cough_real_time'

sample_rate = 44100
n_mfcc = 13
num_classes = 3
frame_length = 1024
frame_step = 512
num_mel_bins = 40
lower_edge_hertz = 0.0
upper_edge_hertz = 8000.0

# Включение GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def extract_features_tf(file_path):
    # Загрузка аудио
    audio = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)
    
    # Нормализация
    audio = audio - tf.math.reduce_mean(audio)
    
    # STFT
    stft = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step)
    spectrogram = tf.abs(stft)
    
    # Mel-спектрограмма
    num_spectrogram_bins = stft.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mel_bins,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=sample_rate,
        lower_edge_hertz=lower_edge_hertz,
        upper_edge_hertz=upper_edge_hertz)
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    
    # Логарифмирование
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    
    # MFCC
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    mfccs = mfccs[..., :n_mfcc]  # Берем только первые n_mfcc коэффициентов
    
    # Усреднение по времени
    mfccs_mean = tf.math.reduce_mean(mfccs, axis=0)
    
    return mfccs_mean.numpy()

# Сбор данных
X = []
y = []

for label in LABELS:
    folder_path = os.path.join(data_dir, label)
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            features = extract_features_tf(file_path)
            X.append(features)
            y.append(LABELS.index(label)) 

X = np.array(X)
y = np.array(y)

# One-hot кодирование
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# Подготовка данных для CNN
X = X.reshape(X.shape[0], 1, n_mfcc, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Создание модели (остается таким же, как у вас)
model = models.Sequential()
model.add(layers.Conv2D(32, (1, 3), activation='relu', 
                       input_shape=(1, n_mfcc, 1), 
                       kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling2D((1, 1)))

model.add(layers.Conv2D(64, (1, 3), activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling2D((1, 1)))

model.add(layers.Conv2D(128, (1, 3), activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling2D((1, 1)))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(0.5)) 
model.add(layers.Dense(num_classes, activation='softmax'))

# Компиляция и обучение
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('cough_detection_model_tfmfcc.h5', 
                           monitor='val_accuracy', 
                           save_best_only=True, 
                           mode='max', 
                           verbose=1)

model.fit(X_train, y_train, epochs=500, batch_size=32, 
         validation_split=0.2, callbacks=[checkpoint])

model.save('cough_detection_model_tfmfcc.h5')
print("Модель сохранена в 'cough_detection_model_tfmfcc.h5'")


# 223/223 [==============================] - ETA: 0s - loss: 0.4394 - accuracy: 0.8693
# Epoch 497: val_accuracy did not improve from 0.88777
# 223/223 [==============================] - 2s 9ms/step - loss: 0.4394 - accuracy: 0.8693 - val_loss: 0.4092 - val_accuracy: 0.8704
# Epoch 498/500
# 222/223 [============================>.] - ETA: 0s - loss: 0.4466 - accuracy: 0.8685
# Epoch 498: val_accuracy did not improve from 0.88777
# 223/223 [==============================] - 2s 9ms/step - loss: 0.4468 - accuracy: 0.8686 - val_loss: 0.4099 - val_accuracy: 0.8771
# Epoch 499/500
# 222/223 [============================>.] - ETA: 0s - loss: 0.4401 - accuracy: 0.8702
# Epoch 499: val_accuracy did not improve from 0.88777
# 223/223 [==============================] - 2s 9ms/step - loss: 0.4398 - accuracy: 0.8703 - val_loss: 0.4083 - val_accuracy: 0.8726
# Epoch 500/500
# 222/223 [============================>.] - ETA: 0s - loss: 0.4359 - accuracy: 0.8698
# Epoch 500: val_accuracy did not improve from 0.88777
# 223/223 [==============================] - 2s 9ms/step - loss: 0.4363 - accuracy: 0.8695 - val_loss: 0.4103 - val_accuracy: 0.8760