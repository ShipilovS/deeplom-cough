# import pyaudio
# import wave
# CHUNK = 1024
# FORMAT = pyaudio.paInt16
# CHANNELS = 2
# RATE = 44100
# RECORD_SECONDS = 5
# WAVE_OUTPUT_FILENAME = "output.wav"
# p = pyaudio.PyAudio()
# stream = p.open(format=FORMAT,
#                 channels=CHANNELS,
#                 rate=RATE,
#                 input=True,
#                 frames_per_buffer=CHUNK)
# print("* recording")
# frames = []
# for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#     data = stream.read(CHUNK)
#     frames.append(data)
# print("* done recording")
# stream.stop_stream()
# stream.close()
# p.terminate()
# wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
# wf.setnchannels(CHANNELS)
# wf.setsampwidth(p.get_sample_size(FORMAT))
# wf.setframerate(RATE)
# wf.writeframes(b''.join(frames))
# wf.close()

import tkinter as tk
import pyaudio
import wave

class AudioRecorder:
    def __init__(self, master):
        self.master = master
        self.master.title("Audio Recorder")

        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        # self.channels = 2
        self.rate = 44100
        self.record_seconds = 5
        self.is_recording = False
        self.output_filename = "output.wav"

        self.p = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        # Input Device id  16  -  default
        # Кнопки
        self.record_button = tk.Button(master, text="Записать", command=self.start_recording)
        self.record_button.pack(pady=10)

        self.stop_button = tk.Button(master, text="Стоп", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(pady=10)

    def start_recording(self):
        self.frames = []
        self.is_recording = True
        self.record_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        self.stream = self.p.open(format=self.format,
                                   channels=self.channels,
                                   rate=int(self.rate),
                                   input=True,
                                   input_device_index=0,
                                   frames_per_buffer=self.chunk)

        print("* recording")
        self.master.after(100, self.record)

    def record(self):
        data = self.stream.read(self.chunk)
        self.frames.append(data)
        if len(self.frames) < int(self.rate / self.chunk * self.record_seconds):
            self.master.after(100, self.record)
        else:
            self.stop_recording()

    def stop_recording(self):
        print("* done recording")
        self.record_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

        # Сохранение в файл
        wf = wave.open(self.output_filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioRecorder(root)
    root.mainloop()
