import tkinter as tk
import sounddevice as sd
import wave
import numpy as np
import matplotlib.pyplot as plt
import threading

# sd.default.hostapi = 'PulseAudio'
# ({'name': 'ALSA', 'devices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 'default_input_device': 16, 'default_output_device': 16}, {'name': 'OSS', 'devices': [], 'default_input_device': -1, 'default_output_device': -1})

class AudioRecorder:
    def __init__(self, master):
        self.master = master
        
        self.master.title("Audio Recorder")

        self.is_recording = False
        self.device_id = None

        # Список доступных устройств
        self.device_list = sd.query_devices()
        self.device_names = [device['name'] for device in self.device_list]

        # Выбор устройства
        self.device_var = tk.StringVar(value=self.device_names[0])
        self.device_menu = tk.OptionMenu(master, self.device_var, *self.device_names)
        self.device_menu.pack(pady=10)

        self.start_button = tk.Button(master, text="Start Recording", command=self.start_recording)
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(master, text="Stop Recording", command=self.stop_recording)
        self.stop_button.pack(pady=10)
        self.stop_button.config(state=tk.DISABLED)

        self.filename = "output.wav"
        self.frames = []
        self.sample_rate = 44100

    def start_recording(self):
        self.is_recording = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        # Получение ID устройства
        self.device_id = self.get_device_id(self.device_var.get())

        # Запуск записи в отдельном потоке
        self.recording_thread = threading.Thread(target=self.record_audio)
        self.recording_thread.start()

    def stop_recording(self):
        self.is_recording = False
        self.stop_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.NORMAL)

        # Ожидание завершения потока записи
        self.recording_thread.join()
        self.save_recording()

    def record_audio(self):
        self.frames = []
        with sd.InputStream(samplerate=int(self.sample_rate), channels=1, device=self.device_id, callback=self.callback, blocksize=2048):
            while self.is_recording:
                print('Записываю')
                sd.sleep(100)  # Небольшая пауза, чтобы не перегружать процессор

    def plot_audio(self):
        if self.frames:
            audio_data = np.concatenate(self.frames, axis=0)
            plt.figure(figsize=(10, 4))
            plt.plot(audio_data)
            plt.title("Записанный звук")
            plt.xlabel("Время (сэмплы)")
            plt.ylabel("Амплитуда")
            plt.grid()
            plt.show()
        else:
            print("Нет записанных данных для отображения.")

    def callback(self, indata, frames, time, status):
        if status:
            print(status)
        if self.is_recording:
            self.frames.append(indata.copy())

    def save_recording(self):
        with wave.open(self.filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 байта для int16
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.frames))
        self.plot_audio()
        print(f"Запись сохранена в {self.filename}")

    def get_device_id(self, device_name):
        for i, device in enumerate(self.device_list):
            if device['name'] == device_name:
                return i
        return None

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioRecorder(root)
    root.mainloop()
