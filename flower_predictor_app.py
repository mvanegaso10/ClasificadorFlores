
import streamlit as st
import joblib
import numpy as np
import os

# --- Cargar el modelo y el codificador de etiquetas ---
# Asegúrate de que las rutas a tus archivos .joblib sean correctas
model_path = '/mlp_model.joblib'
encoder_path = '/label_encoder (1).joblib'

if not os.path.exists(model_path):
    st.error(f"Error: El archivo del modelo no se encuentra en {model_path}")
    st.stop()
if not os.path.exists(encoder_path):
    st.error(f"Error: El archivo del codificador de etiquetas no se encuentra en {encoder_path}")
    st.stop()

mlp_model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)

# --- Configuración de la interfaz de Streamlit ---
st.title('Predicción de Especies de Flores Iris')
st.write('Introduce las características de la flor para predecir su especie.')

# --- Widgets de entrada para las características ---
sepal_length = st.number_input('Longitud del Sépalos (cm)', min_value=0.0, max_value=10.0, value=5.5, step=0.1)
sepal_width = st.number_input('Ancho del Sépalos (cm)', min_value=0.0, max_value=5.0, value=2.5, step=0.1)
petal_length = st.number_input('Longitud del Pétalos (cm)', min_value=0.0, max_value=8.0, value=4.0, step=0.1)
petal_width = st.number_input('Ancho del Pétalos (cm)', min_value=0.0, max_value=3.0, value=1.3, step=0.1)

# --- Botón para realizar la predicción ---
if st.button('Predecir Especie'):
    # Crear un array NumPy con las características ingresadas
    flower_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Realizar la predicción
    predicted_species_encoded = mlp_model.predict(flower_features)

    # La salida del modelo ya está en un formato legible, si el modelo fue entrenado así.
    # Si el modelo predice índices numéricos, necesitarías el label_encoder.
    # Para este caso, asumimos que predicted_species_encoded ya es el nombre de la especie.
    predicted_species = predicted_species_encoded[0]

    st.success(f'La especie de flor predicha es: **{predicted_species}**')

    # Aviso si el modelo usa nombres de características (esto es un warning de scikit-learn)
    # No es un error crítico, pero se puede mencionar si es relevante.
    st.info("Nota: Si ves un mensaje de 'UserWarning' en la consola, es normal. Indica que el modelo fue entrenado con nombres de características pero la predicción se realiza sin ellos explícitamente.")
