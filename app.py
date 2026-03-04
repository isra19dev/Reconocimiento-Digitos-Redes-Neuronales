# Importamos las librerías necesarias
import streamlit as st
import numpy as np
from PIL import Image
import joblib
from streamlit_drawable_canvas import st_canvas

# Configuración de la página
st.set_page_config(page_title="Reconocimiento de Dígitos", layout="centered")

# Centramos el título
col_izq_title, col_title, col_der_title = st.columns([1, 2, 1])
with col_title:
    st.title("🎨 Reconocedor de Dígitos")

# Inicializamos el estado de la sesión
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

# Cargamos el modelo entrenado
@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_digitos.joblib")

# Función para procesar la imagen
def procesar_imagen(canvas_image):
    # Convertimos a escala de grises
    imagen_gris = canvas_image.convert('L')
    
    # Convertimos a array numpy
    imagen_array = np.array(imagen_gris)
    
    # Invertimos la imagen (blanco=0, negro=255)
    imagen_invertida = 255 - imagen_array
    
    # Binarizamos con un threshold menos agresivo
    threshold = 127
    imagen_binaria = np.where(imagen_invertida > threshold, 255, 0)
    
    # Encontramos los píxeles donde hay dígito (valor 255)
    filas = np.any(imagen_binaria == 255, axis=1)
    columnas = np.any(imagen_binaria == 255, axis=0)
    
    # Obtenemos el bounding box del dígito
    if np.any(filas) and np.any(columnas):
        fila_min, fila_max = np.where(filas)[0][[0, -1]]
        col_min, col_max = np.where(columnas)[0][[0, -1]]
        
        # Extraemos la región con el dígito
        digito_region = imagen_binaria[fila_min:fila_max+1, col_min:col_max+1]
    else:
        # Si no hay nada dibujado, devolvemos ceros
        digito_region = np.zeros((8, 8), dtype=np.uint8)
    
    # Redimensionamos a 8x8 manteniendo proporción
    from PIL import Image as PILImage
    digito_pil = PILImage.fromarray(digito_region.astype('uint8'))
    digito_pil.thumbnail((7, 7), PILImage.Resampling.LANCZOS)
    
    # Creamos una imagen 8x8 con el dígito centrado
    imagen_8x8 = np.zeros((8, 8), dtype=np.uint8)
    
    # Calculamos la posición para centrar el dígito
    digito_redimensionado = np.array(digito_pil)
    h, w = digito_redimensionado.shape if len(digito_redimensionado.shape) == 2 else (digito_redimensionado.shape[0], digito_redimensionado.shape[1])
    
    inicio_fila = (8 - h) // 2
    inicio_col = (8 - w) // 2
    
    imagen_8x8[inicio_fila:inicio_fila+h, inicio_col:inicio_col+w] = digito_redimensionado
    
    # Normalizamos los valores de 0-255 a 0-16
    imagen_normalizada = (imagen_8x8 / 255.0) * 16.0
    
    # Aplanamos la imagen a un vector de 64 valores (8x8)
    vector_entrada = imagen_normalizada.flatten()
    
    return vector_entrada

# Centramos el canvas usando columnas
col_izq, col_centro, col_der = st.columns([1, 2, 1])

with col_centro:
    # Canvas para que el usuario dibuje
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=20,
        stroke_color="black",
        background_color="white",
        height=400,
        width=400,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}"
    )

# Espacio para los botones (centrados)
col_izq2, col_btns1, col_btns2, col_der2 = st.columns([0.5, 1, 1, 0.5])

with col_btns1:
    boton_adivinar = st.button("🔍 Adivinar Número", use_container_width=True)

with col_btns2:
    boton_limpiar = st.button("🗑️ Limpiar", use_container_width=True)

# Si el usuario presiona el botón de limpiar
if boton_limpiar:
    st.session_state.canvas_key += 1
    st.rerun()

# Si el usuario presiona el botón de adivinar
if boton_adivinar:
    # Verificamos si hay algo dibujado en el canvas
    if canvas_result.image_data is not None:
        # Obtenemos la imagen dibujada
        imagen_dibujada = Image.fromarray(canvas_result.image_data.astype('uint8'))
        
        # Procesamos la imagen
        vector_entrada = procesar_imagen(imagen_dibujada)
        
        # Cargamos el modelo
        modelo = cargar_modelo()
        
        # Hacemos la predicción
        prediccion = modelo.predict([vector_entrada])[0]
        
        # Mostramos el resultado
        st.success(f"TU NÚMERO ES: {int(prediccion)}")
    else:
        st.warning("Por favor, dibuja un número primero")
