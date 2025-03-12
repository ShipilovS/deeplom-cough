# python3 -m pip install numpy
# python3 main4.py
import tkinter as tk
import sounddevice as sd
import numpy as np
import wave
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import librosa
from tensorflow.keras.models import load_model
# Загрузите вашу модель
# model = load_model('cough_detection_model-37.h5')
from scipy.io.wavfile import write

class AudioRecorder:
    def __init__(self, master):
        self.master = master
        self.master.title("Audio Recorder")

        self.is_recording = False
        self.filename = 'output.wav'
        self.sample_rate = 44100
        self.duration = 5  # Длительность записи в секундах
        self.frames = []

        # Кнопки
        self.record_button = tk.Button(master, text="Записать", command=self.start_recording)
        self.record_button.pack(pady=10)

        self.stop_button = tk.Button(master, text="Стоп", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(pady=10)

        # График
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlim(0, self.sample_rate / 2)
        self.ax.set_title("Уровень звука")
        self.ax.set_xlabel("Частота (Гц)")
        self.ax.set_ylabel("Амплитуда")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.fft_data = []
        self.fft_line, = self.ax.plot([], [], lw=2, color='red')

    def start_recording(self):
        self.is_recording = True
        self.record_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        self.frames = []
        self.thread = threading.Thread(target=self.record)
        self.thread.start()

    def record(self):
        print("* recording")
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32') as stream:
            while self.is_recording:
                print('Записываю')
                data = stream.read(1024)[0]
                self.frames.append(data)
                self.update_plot(data)

    def update_plot(self, data):
        # Преобразование в массив NumPy и вычисление FFT
        fft_data = np.abs(np.fft.fft(data.flatten()))[:1024]
        self.fft_data = fft_data / np.max(fft_data)  # Нормализация

        # Обновление графика
        self.line.set_xdata(np.linspace(0, self.sample_rate / 2, len(self.fft_data)))
        self.line.set_ydata(self.fft_data)
        self.ax.draw_artist(self.line)
        self.canvas.draw_idle()

    def stop_recording(self):
        print("* done recording")
        self.is_recording = False
        self.record_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

        # Сохранение в файл
        self.save_audio()

    # def save_audio(self):
        # output_filename = "output-frames.wav"
        # with open(output_filename, 'wb') as wf:
        #     print(self.frames)
        #     wf.write(b''.join(self.frames))
        # print(f"Запись сохранена в {output_filename}")

    # def save_recording(self):
    #     with wave.open(self.filename, 'wb') as wf:
    #         wf.setnchannels(1)
    #         wf.setsampwidth(2)  # 2 байта для int16
    #         wf.setframerate(self.sample_rate)
    #         wf.writeframes(b''.join(self.frames))

    #     print(f"Запись сохранена в {self.filename}")

    def save_audio(self):
        output_filename = "output-frames.wav"
        # Преобразование данных в одномерный массив
        audio_data = np.concatenate(self.frames).flatten()  # Объединяем и делаем одномерным массивом
        
        # Преобразование в формат int16
        audio_data = (audio_data * 32767).astype(np.int16)

        # Сохранение в WAV файл
        write(output_filename, self.sample_rate, audio_data)  # Используем write из scipy
        print(f"Запись сохранена в {output_filename}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioRecorder(root)
    root.mainloop()
