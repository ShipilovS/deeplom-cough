import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (Dense, InputLayer, Dropout, Flatten, 
                                    Reshape, BatchNormalization, Conv2D, 
                                    MaxPooling2D, AveragePooling2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential

# Параметры
COUGH_DIR = "/home/sshipilov/deeplom_data/Cough"
NO_COUGH_DIR = "/home/sshipilov/deeplom_data/Noize_voise"
SAMPLE_RATE = 44100
N_MFCC = 25
BATCH_SIZE = 32
CHUNK_SIZE = 500  # Размер порции файлов для обработки
MODEL_NAME = 'cough_detection_model_tfmfcc_new_model.h5'

def get_file_chunks():
    """Генератор порций файлов."""
    cough_files = [f.path for f in os.scandir(COUGH_DIR) if f.name.endswith(".wav")]
    no_cough_files = [f.path for f in os.scandir(NO_COUGH_DIR) if f.name.endswith(".wav")]
    
    # Перемешиваем файлы
    np.random.shuffle(cough_files)
    np.random.shuffle(no_cough_files)
    
    # Разбиваем на порции
    for i in range(0, len(cough_files), CHUNK_SIZE):
        chunk_cough = cough_files[i:i+CHUNK_SIZE]
        chunk_no_cough = no_cough_files[i:i+CHUNK_SIZE]
        
        files = chunk_cough + chunk_no_cough
        labels = [1] * len(chunk_cough) + [0] * len(chunk_no_cough)
        
        yield files, labels

def process_files(files, labels):
    """Обработка порции файлов для извлечения MFCC."""
    mfcc_features = []
    processed_labels = []
    
    for file_path, label in zip(files, labels):
        try:
            audio = tf.io.read_file(file_path)
            audio, _ = tf.audio.decode_wav(audio, desired_channels=1)
            audio = tf.squeeze(audio, axis=-1)
            
            # Нормализация
            audio = (audio - tf.reduce_mean(audio)) / (tf.reduce_max(tf.abs(audio)) + 1e-6)
            
            # MFCC
            stft = tf.signal.stft(audio, frame_length=2048, frame_step=512)
            spectrogram = tf.abs(stft)
            
            num_spectrogram_bins = stft.shape[-1]
            linear_to_mel = tf.signal.linear_to_mel_weight_matrix(
                num_mel_bins=40,
                num_spectrogram_bins=num_spectrogram_bins,
                sample_rate=SAMPLE_RATE,
                lower_edge_hertz=20,
                upper_edge_hertz=8000
            )
            
            mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel, 1)
            log_mel = tf.math.log(mel_spectrogram + 1e-6)
            mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)[..., :N_MFCC]
            
            # Усреднение по времени
            mfccs_mean = tf.reduce_mean(mfccs, axis=0)
            
            mfcc_features.append(mfccs_mean.numpy())
            processed_labels.append(label)
        except Exception as e:
            print(f"Ошибка обработки файла {file_path}: {str(e)}")
            continue
    
    X = np.array(mfcc_features)
    y = np.array(processed_labels)
    
    # Преобразование формы для Conv2D
    X = X.reshape(X.shape[0], N_MFCC, 1)  # Форма: (n_samples, N_MFCC, 1)

    return X, y


def create_model():
    """Создание модели для классификации с использованием свёрточных слоев."""
    model = models.Sequential([
        layers.Input(shape=(N_MFCC, 1)),  # Изменена форма входа
        layers.Conv2D(32, (3, 1), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 1)),
        layers.Conv2D(64, (3, 1), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 1)),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    """Обучение модели."""
    # Создаем или загружаем модель
    if os.path.exists(MODEL_NAME):
        model = models.load_model(MODEL_NAME)
        print("Загружена существующая модель для дообучения")
    else:
        model = create_model()
        print("Создана новая модель")
    
    # Определяем колбеки
    callbacks = [
        ModelCheckpoint(MODEL_NAME, save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')
    ]
    i = 1
    # Обрабатываем данные порциями
    for chunk_num, (files, labels) in enumerate(get_file_chunks()):
        print(f"\nОбработка порции {chunk_num + 1} ({len(files)} файлов)")
        
        # Разделение на train/val с сохранением баланса классов
        train_files, val_files, train_labels, val_labels = train_test_split(
            files, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Обработка
        X_train, y_train = process_files(train_files, train_labels)
        X_val, y_val = process_files(val_files, val_labels)
        
        # Обучение
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=500,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
            workers=16,
            use_multiprocessing=True
        )
        
        # Сохраняем модель после каждой порции
        model.save(f'cough_detection_model_tfmfcc_new_model_{i}.h5')
        i += 1
        print(f"Точность на валидации: {history.history['val_accuracy'][-1]:.4f}")

# Запуск обучения
if __name__ == "__main__":
    train_model()

