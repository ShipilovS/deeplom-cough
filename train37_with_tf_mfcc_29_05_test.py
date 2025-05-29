import os
import numpy as np
import tensorflow as tf

# Пути к папкам с аудиофайлами
COUGH_DIR = "/home/sshipilov/deeplom_data/Cough"
NO_COUGH_DIR = "/home/sshipilov/deeplom_data/Noize_voise"
SAMPLE_RATE = 22050
MODEL_NAME = 'cough_detection_model_tfmfcc_new_model_48.h5'

def load_and_process_audio_file(file_path):
    """Загрузить и обработать один аудиофайл."""
    try:
        audio = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio, desired_channels=1)
        audio = tf.squeeze(audio, axis=-1)
        
        # Нормализация
        audio = (audio - tf.reduce_mean(audio)) / (tf.reduce_max(tf.abs(audio)) + 1e-6)
        
        # Обеспечиваем длину SAMPLE_RATE
        if len(audio) < SAMPLE_RATE:
            audio = np.pad(audio, (0, SAMPLE_RATE - len(audio)), mode='constant')
        else:
            audio = audio[:SAMPLE_RATE]
        
        return audio.numpy()
    except Exception as e:
        print(f"Ошибка при загрузке файла {file_path}: {str(e)}")
        return None

def predict_files(model, directory, label, num_files=5):
    """Предсказать классы для последних num_files файлов из директории."""
    files = [f for f in os.listdir(directory) if f.endswith(".wav")]
    files = sorted(files)[-num_files:]  # берем последние num_files файлов
    
    for filename in files:
        file_path = os.path.join(directory, filename)
        audio = load_and_process_audio_file(file_path)
        
        if audio is not None:  # Проверяем, успешно ли загружен файл
            # Подготовка формы для модели
            x = audio.reshape(1, 1, SAMPLE_RATE, 1)
            prediction = model.predict(x)
            pred_class = np.argmax(prediction, axis=1)[0]
            print(f"Файл: {filename} | Истинный класс: {label} | Предсказанный класс: {pred_class}")

# Загрузка модели
model = tf.keras.models.load_model(MODEL_NAME)

print("Предсказания для последних файлов с кашлем:")
predict_files(model, COUGH_DIR, label=1, num_files=20)

print("\nПредсказания для последних файлов без кашля:")
predict_files(model, NO_COUGH_DIR, label=0, num_files=20)
