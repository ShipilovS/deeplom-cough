import numpy as np
import librosa
from tensorflow.keras.models import load_model

LABELS = ['VGMU_Coswata_Cutt', 'VGMU_Noise', 'VGMU_Voises_Cutt']

# Загрузка модели
model = load_model('cough_detection_model-37.h5')

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# Укажите путь к вашему тестовому аудиофайлу
# test_file_path = './VGMU_Coswata_Cutt/8cough-heavy.wav' 
test_file_path = './VGMU_Noise/1-977-A-39.wav' 
# test_file_path = './VGMU_Voises_Cutt/4630dec50de21d73a8b9966a246c4a39.wav'

features = extract_features(test_file_path)

features = features.reshape(1, 1, 13, 1) 

predictions = model.predict(features)
predicted_class = np.argmax(predictions)

print(f"Предсказанный класс: {predicted_class} ({LABELS[predicted_class]})")
