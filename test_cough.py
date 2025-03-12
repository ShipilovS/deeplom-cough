import numpy as np
import sounddevice as sd
import tensorflow as tf

# Загрузка модели
model = tf.keras.models.load_model('cough_detection_model.h5')

# Параметры записи
sample_rate = 22050  # Частота дискретизации
block_size = 2205  # Размер блока (примерно 0.1 секунды)

def callback(indata, frames, time, status):
    if status:
        print(status)

    # Предобработка аудио
    audio_data = indata.flatten()  # Преобразование в одномерный массив
    audio_data = audio_data / np.max(np.abs(audio_data))  # Нормализация

    # Изменение размера для модели (1, длина, 1)
    input_data = audio_data.reshape(1, -1, 1)

    # Предсказание
    prediction = model.predict(input_data)
    if prediction[0] > 0.5:  # Предположим, что модель возвращает вероятность
        print("Это кашель!")
    else:
        print("Это не кашель.")

# Запуск потока
with sd.InputStream(callback=callback, channels=1, samplerate=sample_rate, blocksize=block_size):
    print("Запись в реальном времени... Нажмите Ctrl+C для остановки.")
    try:
        while True:
            sd.sleep(1000)  # Поддерживаем поток активным
    except KeyboardInterrupt:
        print("Запись остановлена.")
