# predictor.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from train_model import train_model  # Importa la función train_model() del archivo train_model.py

def cargar_modelo():
    model = tf.keras.models.load_model('modelo_entrenado.h5')
    return model

def predecir_malware(ruta_imagen, model):
    img = image.load_img(ruta_imagen, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalizar
    prediction = model.predict(img_array)
    if prediction[0][0] >= 0.5: # Asumiendo que la clase 0 corresponde a malware
        return "La imagen es malware."
    else:
        return "La imagen no es malware."

if __name__ == "__main__":
    # Cargar el modelo entrenado
    modelo_entrenado = cargar_modelo()
    print(modelo_entrenado)
    # Ruta de la imagen a predecir
    ruta_imagen_a_predecir = r"C:\Users\zapat\OneDrive\Documentos\DeepLearning\save_img_converted\000bde2e9a94ba41c0c111ffd80647c2.png"
    resultado = predecir_malware(ruta_imagen_a_predecir, modelo_entrenado)
    print(resultado)
