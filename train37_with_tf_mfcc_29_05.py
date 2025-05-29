import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.optimizers import Adam
# from kapre.time_frequency import Melspectrogram
from kapre.augmentation import AdditiveNoise
from tensorflow.keras.layers import MelSpectrogram

# Параметры
COUGH_DIR = "/home/sshipilov/deeplom_data/Cough"
NO_COUGH_DIR = "/home/sshipilov/deeplom_data/Noize_voise"
SAMPLE_RATE = 22050
N_MFCC = 25
BATCH_SIZE = 32
CHUNK_SIZE = 1000
MODEL_NAME = 'cough_detection_model_tfmfcc_new_model.h5'

# Включение GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def get_file_chunks():
    """Генератор для чанков файлов."""
    cough_files = [f.path for f in os.scandir(COUGH_DIR) if f.name.endswith(".wav")]
    no_cough_files = [f.path for f in os.scandir(NO_COUGH_DIR) if f.name.endswith(".wav")]
    
    np.random.shuffle(cough_files)
    np.random.shuffle(no_cough_files)
    
    for i in range(0, len(cough_files), CHUNK_SIZE):
        chunk_cough = cough_files[i:i+CHUNK_SIZE]
        chunk_no_cough = no_cough_files[i:i+CHUNK_SIZE]
        
        files = chunk_cough + chunk_no_cough
        labels = [1] * len(chunk_cough) + [0] * len(chunk_no_cough)
        
        yield files, labels

def process_files(files, labels):
    """Обработка батча файлов для извлечения сырых аудиосигналов."""
    audio_features = []
    processed_labels = []
    
    for file_path, label in zip(files, labels):
        try:
            audio = tf.io.read_file(file_path)
            audio, _ = tf.audio.decode_wav(audio, desired_channels=1)
            audio = tf.squeeze(audio, axis=-1)
            
            # Нормализация
            audio = (audio - tf.reduce_mean(audio)) / (tf.reduce_max(tf.abs(audio)) + 1e-6)
            
            # Обеспечьте длину 44100
            if len(audio) < SAMPLE_RATE:
                audio = np.pad(audio, (0, SAMPLE_RATE - len(audio)), mode='constant')
            else:
                audio = audio[:SAMPLE_RATE]
            
            # Добавляем дополнительное измерение
            audio_features.append(audio.numpy()[np.newaxis, :])  # Теперь форма (1, 44100)
            processed_labels.append(label)
        except Exception as e:
            print(f"Ошибка обработки файла {file_path}: {str(e)}")
            continue
    
    X = np.array(audio_features)
    y = np.array(processed_labels)

    # One-hot кодирование
    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y.reshape(-1, 1))

    # Изменение формы для Conv2D
    X = X.reshape(X.shape[0], 1, SAMPLE_RATE, 1)  # Форма: (n_samples, 1, 44100, 1)
    print(X.shape)
    return X, y


def create_model():
    model = models.Sequential()

    # Вход: аудио сигнал длиной SAMPLE_RATE (44100)
    model.add(layers.Input(shape=(SAMPLE_RATE, 1)))

    # MelSpectrogram преобразует (batch, samples) -> (batch, time_frames, mel_bins, 1)
    model.add(layers.MelSpectrogram(
        fft_length=2048,
        sequence_length=512,  
        window='hann',
        sampling_rate=SAMPLE_RATE,
        num_mel_bins=32,
        min_freq=10.0,
        max_freq=8000.0,
        power_to_db=True,
        top_db=80.0,
        mag_exp=2.0,
        min_power=1e-10,
        ref_power=1.0,
    ))

    # Добавляем размерность канала для Conv2D
    model.add(layers.Reshape((-1, 32, 1)))  # Автоматический расчет временной размерности

    # Сверточные слои
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                           kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu', 
                          kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(optimizer='adam', 
                 loss='categorical_crossentropy', 
                 metrics=['accuracy'])
    model.summary()
    
    return model

def train_model():
    """Обучение модели."""
    if os.path.exists(MODEL_NAME):
        model = models.load_model(MODEL_NAME)
        print("Загружена существующая модель для дообучения")
    else:
        model = create_model()
        print("Создана новая модель")
    
    callbacks = [
        ModelCheckpoint(MODEL_NAME, save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
    ]
    
    for chunk_num, (files, labels) in enumerate(get_file_chunks()):
        print(f"\nОбработка чанка {chunk_num + 1} ({len(files)} файлов)")
        
        train_files, val_files, train_labels, val_labels = train_test_split(
            files, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        X_train, y_train = process_files(train_files, train_labels)
        X_val, y_val = process_files(val_files, val_labels)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
        )
        
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        print(f"Потеря на валидации: {val_loss:.4f}, Точность на валидации: {val_accuracy:.4f}")

        model.save(f'cough_detection_model_tfmfcc_new_model_{chunk_num + 1}.h5')
        print(f"Точность на валидации из истории: {history.history['val_accuracy'][-1]:.4f}")

if __name__ == "__main__":
    train_model()