import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, InputLayer, Dropout, Flatten, 
                                    Reshape, BatchNormalization, Conv2D, 
                                    MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Параметры
COUGH_DIR = "/home/sshipilov/deeplom_data/Cough"
NO_COUGH_DIR = "/home/sshipilov/deeplom_data/Noize_voise"
SAMPLE_RATE = 44100
N_MFCC = 26  # Должно делиться на 13 для Reshape
BATCH_SIZE = 64
CHUNK_SIZE = 1000
MODEL_NAME = 'conv_cough_detection_model.h5'
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_and_preprocess_data():
    """Загрузка и предварительная обработка данных"""
    # Сбор и разделение файлов
    cough_files = [f.path for f in os.scandir(COUGH_DIR) if f.name.endswith(".wav")]
    no_cough_files = [f.path for f in os.scandir(NO_COUGH_DIR) if f.name.endswith(".wav")]
    
    # Разделение на train/test
    cough_train, cough_test = train_test_split(cough_files, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    no_cough_train, no_cough_test = train_test_split(no_cough_files, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # Формирование итоговых наборов
    train_files = cough_train + no_cough_train
    train_labels = [1]*len(cough_train) + [0]*len(no_cough_train)
    
    test_files = cough_test + no_cough_test
    test_labels = [1]*len(cough_test) + [0]*len(no_cough_test)
    
    # Перемешивание
    train_data = list(zip(train_files, train_labels))
    np.random.shuffle(train_data)
    return [f for f, _ in train_data], [l for _, l in train_data], test_files, test_labels

def extract_mfcc_features(file_path):
    """Извлечение MFCC признаков из аудиофайла"""
    audio = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)
    
    # Нормализация
    audio = audio - tf.math.reduce_mean(audio)
    audio = audio / (tf.math.reduce_max(tf.abs(audio)) + 1e-6)
    
    # Вычисление MFCC
    stft = tf.signal.stft(audio, frame_length=2048, frame_step=512)
    spectrogram = tf.abs(stft)
    
    linear_to_mel = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=40,
        num_spectrogram_bins=stft.shape[-1],
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=20,
        upper_edge_hertz=8000
    )
    
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel, 1)
    log_mel = tf.math.log(mel_spectrogram + 1e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)[..., :N_MFCC]
    
    # Усреднение по времени и нормализация
    mfccs_mean = tf.reduce_mean(mfccs, axis=0)
    mfccs_norm = (mfccs_mean - tf.reduce_mean(mfccs_mean)) / tf.math.reduce_std(mfccs_mean)
    
    return mfccs_norm.numpy()

def create_conv_model(input_shape):
    # """Создание CNN модели на основе вашей архитектуры"""
    # model = Sequential([
    #     InputLayer(input_shape=(input_shape,), name='input_layer'),
    #     Reshape((input_shape//13, 13, 1)),  # Преобразуем в 2D (временные кадры × коэффициенты MFCC × 1 канал)
        
    #     Conv2D(10, kernel_size=5, activation='relu', padding='same', kernel_constraint=MaxNorm(3)),
    #     BatchNormalization(),
    #     AveragePooling2D(pool_size=2, padding='same'),
    #     Dropout(0.3),
        
    #     Conv2D(5, kernel_size=5, activation='relu', padding='same', kernel_constraint=MaxNorm(3)),
    #     BatchNormalization(),
    #     AveragePooling2D(pool_size=2, padding='same'),
    #     Dropout(0.3),
        
    #     Flatten(),
    #     Dense(64, activation='relu', kernel_constraint=MaxNorm(3)),
    #     Dropout(0.5),
    #     Dense(1, activation='sigmoid', name='output_layer')
    # ])
    
    # optimizer = Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999)
    # model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    # return model
    
    model = Sequential([
        InputLayer(input_shape=(input_shape,), name='input_layer'),
        Reshape((input_shape // 13, 13, 1)),
        
        Conv2D(32, kernel_size=5, activation='relu', padding='same', kernel_constraint=MaxNorm(3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=2, padding='same'),
        Dropout(0.5),
        
        Conv2D(64, kernel_size=5, activation='relu', padding='same', kernel_constraint=MaxNorm(3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=2, padding='same'),
        Dropout(0.5),
        
        Flatten(),
        Dense(128, activation='relu', kernel_constraint=MaxNorm(3)),
        Dropout(0.5),
        Dense(1, activation='sigmoid', name='output_layer')
    ])
    
    
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

def train_in_batches():
    """Обучение модели порциями с сохранением после каждой итерации"""
    train_files, train_labels, test_files, test_labels = load_and_preprocess_data()
    
    # Сначала обрабатываем один файл для определения размерности
    sample_feature = extract_mfcc_features(train_files[0])
    input_shape = sample_feature.shape[0]
    
    # Проверяем, что MFCC можно разделить на 13 для Reshape
    if input_shape % 13 != 0:
        raise ValueError(f"N_MFCC должно делиться на 13. Текущее значение: {input_shape}")
    
    model = create_conv_model(input_shape)
    
    # Создаем папку для сохранения моделей, если ее нет
    os.makedirs('saved_models', exist_ok=True)
    
    best_val_accuracy = 0
    best_model_path = ''
    
    # Обучение порциями
    for i in range(0, len(train_files), CHUNK_SIZE):
        chunk_files = train_files[i:i+CHUNK_SIZE]
        chunk_labels = train_labels[i:i+CHUNK_SIZE]
        
        print(f"\nProcessing batch {i//CHUNK_SIZE + 1}...")
        
        # Извлечение признаков
        features = [extract_mfcc_features(f) for f in chunk_files]
        X_batch = np.array(features)
        y_batch = np.array(chunk_labels)
        
        # Разделение на train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X_batch, y_batch, test_size=0.2, random_state=RANDOM_STATE
        )
        
        # Callback для сохранения лучшей модели в этой итерации
        iteration_model_path = f'saved_models/model_iteration_{i//CHUNK_SIZE + 1}.h5'

        callbacks = [
            ModelCheckpoint(iteration_model_path, 
                          monitor='val_accuracy', 
                          save_best_only=True, 
                          mode='max',
                          save_weights_only=False),
            EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=BATCH_SIZE,
            epochs=25,
            callbacks=callbacks,
            verbose=2
        )
        
        # Сохраняем модель после каждой итерации
        current_iteration_model_path = f'saved_models/model_after_iteration_{i//CHUNK_SIZE + 1}.h5'
        model.save(current_iteration_model_path)
        print(f"Model saved to {current_iteration_model_path}")
        
        # Проверяем, не является ли текущая модель лучшей
        current_val_accuracy = max(history.history['val_accuracy'])
        if current_val_accuracy > best_val_accuracy:
            best_val_accuracy = current_val_accuracy
            best_model_path = f'saved_models/best_model_iteration_{i//CHUNK_SIZE + 1}.h5'
            model.save(best_model_path)
            print(f"New best model saved to {best_model_path} with val_accuracy: {best_val_accuracy:.4f}")
    
    # Сохраняем финальную модель
    final_model_path = 'saved_models/final_model.h5'
    model.save(final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    
    # Оценка на тестовых данных
    test_features = [extract_mfcc_features(f) for f in test_files]
    X_test = np.array(test_features)
    y_test = np.array(test_labels)
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
    
    # Загружаем и оцениваем лучшую модель
    if best_model_path:
        print(f"\nEvaluating the best model from {best_model_path}")
        best_model = tf.keras.models.load_model(best_model_path)
        loss, accuracy = best_model.evaluate(X_test, y_test, verbose=0)
        print(f"Best Model Test Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")

if __name__ == "__main__":
    train_in_batches()