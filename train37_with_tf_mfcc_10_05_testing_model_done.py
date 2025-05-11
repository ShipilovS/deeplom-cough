import os
import random
import numpy as np
import tensorflow as tf

# Параметры
COUGH_DIR = "/home/sshipilov/deeplom_data/Cough"
NO_COUGH_DIR = "/home/sshipilov/deeplom_data/Noize_voise"
SAMPLE_RATE = 44100
N_MFCC = 25
MODEL_NAME = 'cough_detection_model_tfmfcc_new_model.h5'

# Функция для загрузки и обработки аудио файла
def load_and_process_audio(file_path):
    audio = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)
    audio = audio - tf.math.reduce_mean(audio)

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
    mfccs_mean = tf.reduce_mean(mfccs, axis=0)

    # Изменение формы MFCC для соответствия (1, 1, N_MFCC, 1)
    # return mfccs_mean.numpy().reshape(1, 1, N_MFCC, 1)
    return mfccs_mean.numpy().reshape(1, N_MFCC)  # Изменяем здесь


def evaluate_model_on_random_files(model, threshold=0.5):
    # Случайный выбор 1 файла из каждой директории
    cough_files = [f.path for f in os.scandir(COUGH_DIR) if f.name.endswith(".wav")]
    no_cough_files = [f.path for f in os.scandir(NO_COUGH_DIR) if f.name.endswith(".wav")]

    selected_files = random.sample(cough_files, 1) + random.sample(no_cough_files, 1)
    
    for file_path in selected_files:
        # file_path = '/home/sshipilov/deeplom_data/Cough/2ef9e67b3-9147-494f-a855-17f2326a61f3.wav' # на самомо деле там не кашель
        mfcc_features = load_and_process_audio(file_path)
        prediction = model.predict(mfcc_features)

        # Выводим форму предсказания для отладки
        print(f"Форма предсказания: {prediction.shape} - {prediction}")

        # Вероятность кашля (предполагаем, что prediction[0][0] - это вероятность кашля)
        cough_probability = prediction[0][0]

        # Используем заданный порог
        predicted_label = 1 if cough_probability > threshold else 0
        true_label = 1 if file_path in cough_files else 0
        
        print(f"Файл: {file_path}")
        print(f"Предсказанный класс: {'Кашель' if predicted_label == 1 else 'Нет кашля'}")
        print(f"Вероятность кашля: {cough_probability:.4f}")  # Выводим вероятность кашля
        print(f"Истинный класс: {'Кашель' if true_label == 1 else 'Нет кашля'}")
        print("-" * 40)




# Загрузка модели
model = tf.keras.models.load_model(MODEL_NAME)

# Оценка модели на случайных файлах
evaluate_model_on_random_files(model)
evaluate_model_on_random_files(model)
evaluate_model_on_random_files(model)
evaluate_model_on_random_files(model)
evaluate_model_on_random_files(model)
evaluate_model_on_random_files(model)
evaluate_model_on_random_files(model)
evaluate_model_on_random_files(model)
evaluate_model_on_random_files(model)


# Форма предсказания: (1, 1) - [[0.99556446]]
# Файл: /home/sshipilov/deeplom_data/Noize_voise/3905149400db1-3ca3-472d-b4ba-aef8261d7712.wav
# ну тут шум просто фраза - это тоже надо
# Предсказанный класс: Кашель
# Вероятность кашля: 0.9956
# Истинный класс: Нет кашля
# ------------------------------

# 1/1 [==============================] - 0s 21ms/step
# Форма предсказания: (1, 1) - [[0.00017914]]
# Файл: /home/sshipilov/deeplom_data/Cough/2ef9e67b3-9147-494f-a855-17f2326a61f3.wav
# Предсказанный класс: Нет кашля
# Вероятность кашля: 0.0002
# Истинный класс: Кашель
