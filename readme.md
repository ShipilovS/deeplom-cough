```bash
    sudo apt-get install python3-tk
```

```bash
    sudo apt install pulseaudio pulseaudio-utils
```


```bash
    sudo apt install alsa-base alsa-utils
```



sudo chmod -R 777 main.py

sudo apt-get install python-tk
sudo apt-get install python3.7-tk

import sounddevice as sd
python3 -m pip install numpy --upgrade --use-deprecated=legacy-resolver
print(sd.query_hostapis())
print(sd.query_devices())

ldconfig -p | grep cudnn
ldconfig -p | grep cuda


import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

nvidia-smi -l 1 


для train37.py можено или и 

```python

X = np.array(X)
y = np.array(y)

# encoder = OneHotEncoder(sparse_output=False) # для версии python-3.10
encoder = OneHotEncoder(sparse=False)# для версии python-3.7
y = encoder.fit_transform(y.reshape(-1, 1))

X = X.reshape(X.shape[0], 1, X.shape[1], 1)
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
                             
print(f"train")
# Создание модели
model = models.Sequential()
model.add(layers.Conv2D(32, (1, 3), activation='relu', 
                         input_shape=(1, 13, 1), kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling2D((1, 2)))

model.add(layers.Conv2D(64, (1, 3), activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling2D((1, 2)))
model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(0.5)) 
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('cough_detection_model-37.h5', 
                             monitor='val_accuracy', 
                             save_best_only=True, 
                             mode='max', 
                             verbose=1)

model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.2, callbacks=[checkpoint])

model.save('cough_detection_model-37.h5')
print("Модель сохранена в 'cough_detection_model.h5'")

Запускать - train37.py
