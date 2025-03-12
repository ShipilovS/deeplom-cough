import tensorflow as tf

# Проверка доступных устройств
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Доступные GPU: {gpus}")
else:
    print("GPU не найдено.")