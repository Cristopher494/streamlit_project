import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

# Cargar el modelo
model = load_model('../models/cats_vs_dogs-densenet121_model.h5')  # Reemplaza con la ruta a tu modelo

# Título de la aplicación
st.title('Clasificación de Imágenes de Perros y Gatos')

st.write("""
## Descripción del Proyecto
Este proyecto utiliza un modelo de aprendizaje profundo para clasificar imágenes de perros y gatos. 
Puedes subir una imagen de tu perro o gato, y el modelo intentará predecir correctamente si es un perro o un gato.
El modelo fue entrenado usando una red neuronal convolucional (CNN) y está optimizado para proporcionar predicciones precisas.
""")

# Incluir una imagen de perro o gato en la página principal
st.image('sample.png', width=200)

# Subir imagen
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar la imagen subida
    img = Image.open(uploaded_file)
    st.image(img, caption='Imagen subida', use_column_width=True)
    
    # Preparar la imagen para la predicción
    img = img.resize((224, 224))  # Reemplaza con el tamaño que tu modelo espera
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalizar según sea necesario

    # Hacer la predicción
    prediction = model.predict(img_array)
    class_names = ['Gato', 'Perro']  # Asegúrate de que estas clases coincidan con tu modelo
    
    # Mostrar la predicción
    st.write(f"Predicción: {class_names[np.argmax(prediction)]}")