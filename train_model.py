import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def train_model(data_dir, img_width, img_height, train_split):
    # Preprocesamiento y aumento de datos
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=1-train_split) # 20% de las imágenes se usarán para validación

    # Carga de las imágenes desde el directorio y división en conjuntos de entrenamiento y validación
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='categorical',
        subset='training') # Conjunto de entrenamiento

    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='categorical',
        subset='validation') # Conjunto de validación
    
    # Definición del modelo CNN
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(25, activation='softmax')
    ])

     # Compilación del modelo
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Entrenamiento del modelo
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=8,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size
    )
    # Guardar el modelo entrenado
    model.save('modelo_entrenado.h5')

    return model

if __name__ == "__main__":
    data_dir = r"C:\Users\zapat\OneDrive\Documentos\cnn\malimg_dataset\malimg_paper_dataset_imgs"
    img_width, img_height = 150, 150
    train_split = 0.8
    train_model(data_dir, img_width, img_height, train_split)