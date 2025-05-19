import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder  # Import OneHotEncoder
from tensorflow.keras.layers import (Dense, InputLayer, Dropout, Flatten, 
                                    Reshape, BatchNormalization, Conv2D, 
                                    MaxPooling2D, AveragePooling2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential

# Parameters
COUGH_DIR = "/home/sshipilov/deeplom_data/Cough"
NO_COUGH_DIR = "/home/sshipilov/deeplom_data/Noize_voise"
SAMPLE_RATE = 44100
N_MFCC = 25
BATCH_SIZE = 32
CHUNK_SIZE = 500  # Size of file chunks for processing
MODEL_NAME = 'cough_detection_model_tfmfcc_new_model.h5'

# Включение GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def get_file_chunks():
    """Generator for file chunks."""
    cough_files = [f.path for f in os.scandir(COUGH_DIR) if f.name.endswith(".wav")]
    no_cough_files = [f.path for f in os.scandir(NO_COUGH_DIR) if f.name.endswith(".wav")]
    
    # Shuffle files
    np.random.shuffle(cough_files)
    np.random.shuffle(no_cough_files)
    
    # Split into chunks
    for i in range(0, len(cough_files), CHUNK_SIZE):
        chunk_cough = cough_files[i:i+CHUNK_SIZE]
        chunk_no_cough = no_cough_files[i:i+CHUNK_SIZE]
        
        files = chunk_cough + chunk_no_cough
        labels = [1] * len(chunk_cough) + [0] * len(chunk_no_cough)
        
        yield files, labels

def process_files(files, labels):
    """Process a batch of files to extract MFCC."""
    mfcc_features = []
    processed_labels = []
    
    for file_path, label in zip(files, labels):
        try:
            audio = tf.io.read_file(file_path)
            audio, _ = tf.audio.decode_wav(audio, desired_channels=1)
            audio = tf.squeeze(audio, axis=-1)
            
            # Normalize
            audio = (audio - tf.reduce_mean(audio)) / (tf.reduce_max(tf.abs(audio)) + 1e-6)
            
            # MFCC
            stft = tf.signal.stft(audio, frame_length=2048, frame_step=512)
            spectrogram = tf.abs(stft)
            
            num_spectrogram_bins = stft.shape[-1]
            linear_to_mel = tf.signal.linear_to_mel_weight_matrix(
                num_mel_bins=40,
                num_spectrogram_bins=num_spectrogram_bins,
                sample_rate=SAMPLE_RATE,
                lower_edge_hertz=20,
                upper_edge_hertz=8000
            )
            
            mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel, 1)
            log_mel = tf.math.log(mel_spectrogram + 1e-6)
            mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)[..., :N_MFCC]
            
            # Average over time
            mfccs_mean = tf.reduce_mean(mfccs, axis=0)
            
            mfcc_features.append(mfccs_mean.numpy())
            processed_labels.append(label)
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            continue
    
    X = np.array(mfcc_features)
    y = np.array(processed_labels)
    
    # One-hot encoding
    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(y.reshape(-1, 1))

    # Reshape for Conv2D
    X = X.reshape(X.shape[0], 1, N_MFCC, 1)  # Shape: (n_samples, 1, N_MFCC, 1)

    return X, y

def create_model():
    """Create a model for classification using convolutional layers."""
    model = models.Sequential()

    # Первый сверточный слой
    model.add(layers.Conv2D(
        filters=32,
        kernel_size=(1, 3),
        activation='relu',
        input_shape=(1, N_MFCC, 1),
        kernel_regularizer=regularizers.l2(0.01)
    ))
    model.add(layers.Dropout(0.5))  # Dropout для регуляризации и борьбы с переобучением
    model.add(layers.MaxPooling2D(pool_size=(1, 1)))  # Пулинг по размеру 1x1 (фактически не уменьшает размер)

    # Второй сверточный слой
    model.add(layers.Conv2D(
        filters=64,
        kernel_size=(1, 3),
        activation='relu'
    ))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D(pool_size=(1, 1)))

    # Третий сверточный слой
    model.add(layers.Conv2D(
        filters=128,
        kernel_size=(1, 3),
        activation='relu'
    ))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D(pool_size=(1, 1)))

    # Выравнивание тензора перед полносвязными слоями
    model.add(layers.Flatten())

    # Полносвязный слой с регуляризацией
    model.add(layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.01)
    ))
    model.add(layers.Dropout(0.5))

    # Выходной слой с softmax — для классификации на num_classes классов
    model.add(layers.Dense(2, activation='softmax'))

    # Вывод структуры модели
    model.summary()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',  # Changed to categorical_crossentropy
        metrics=['accuracy']
    )
    
    return model

def train_model():
    """Train the model."""
    # Create or load the model
    if os.path.exists(MODEL_NAME):
        model = models.load_model(MODEL_NAME)
        print("Loaded existing model for fine-tuning")
    else:
        model = create_model()
        print("Created new model")
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint(MODEL_NAME, save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
    ]
    i = 1
    # Process data in chunks
    for chunk_num, (files, labels) in enumerate(get_file_chunks()):
        print(f"\nProcessing chunk {chunk_num + 1} ({len(files)} files)")
        
        # Split into train/val while preserving class balance
        train_files, val_files, train_labels, val_labels = train_test_split(
            files, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Processing
        X_train, y_train = process_files(train_files, train_labels)
        X_val, y_val = process_files(val_files, val_labels)
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=500,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
            workers=16,
            use_multiprocessing=True
        )
        
        # Evaluate the model on validation data after training on this chunk
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Save model after each chunk
        model.save(f'cough_detection_model_tfmfcc_new_model_{i}.h5')
        i += 1
        print(f"Validation accuracy from history: {history.history['val_accuracy'][-1]:.4f}")

# Start training
if __name__ == "__main__":
    train_model()
