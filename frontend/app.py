import streamlit as st
import requests
import pandas as pd
from utils import list_tables
from pathlib import Path
import time
import librosa
import librosa.display
import matplotlib.pyplot as plt

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Songs Recommender", layout="wide", page_icon="🎧")

st.sidebar.write("Hola!👋 Te ayudaré a encontrar canciones similares a las que tienes en mente.")
page = st.sidebar.radio("Selecciona una opción:", [
    "🔍 Consulta SQL",
    "🎧 Buscar por Audio",
    "📁 Gestión de Tablas"
])

st.markdown("""
    <style>
        .block-container { padding-top: 2rem; }
        .stRadio > div { flex-direction: row; }
        .stDataFrame { max-height: 400px; overflow-y: auto; }
    </style>
""", unsafe_allow_html=True)

st.title("🎧 Songs Intelligent Recommender System")

if page == "🔍 Consulta SQL":

# --------------------------------- CONSULTA SQL -------------------------------
    available_tables = list_tables("data")
    selected_table = st.selectbox("📂 Elige la tabla:", available_tables, index=available_tables.index("Audio") if "Audio" in available_tables else 0)

    metadata_path = Path("data") / selected_table / "metadata.csv"
    try:
        metadata_df = pd.read_csv(metadata_path, nrows=1)  # Solo primera fila para mostrar columnas
        columns = metadata_df.columns.tolist()
        st.info(f"🧩 Features disponibles en **{selected_table}**:\n\n`{', '.join(columns)}`")

        default_query = f"SELECT track_name, track_artist FROM {selected_table} WHERE lyrics LIKE 'love' LIMIT 5;"
        user_query = st.text_area("Run your query:", default_query, height=100)

        if st.button("Execute"):
            with st.spinner("Procesando..."):
                print("Query:", user_query)
                res = requests.post(f"{API_URL}/search_sql", json={"query": user_query})

                start_time = time.time()
                res = requests.post(f"{API_URL}/search_sql", json={"query": user_query})
                elapsed_ms = round((time.time() - start_time) * 1000, 2)

                if res.status_code == 200:
                    data = res.json()
                    if data:
                        df = pd.DataFrame(data)
                        st.dataframe(df)

                        # Mostrar tiempo en gris
                        st.markdown(f"<div style='color: gray;'>⏱️ Tiempo de ejecución: {elapsed_ms} ms</div>", unsafe_allow_html=True)
                    else:
                        st.warning("No se encontraron resultados.")
                else:
                    st.error("Error al interpretar la consulta.")
    except Exception as e:
        st.error(f"Error al leer metadatos: {e}")

# --------------------------------- BUSQUEDA POR AUDIO -------------------------------

elif page == "🎧 Buscar por Audio":

    st.subheader("🎤 Buscar canciones similares por audio")
    uploaded_file = st.file_uploader("🔊 Carga un archivo (.mp3 o .wav)", type=["mp3", "wav"])
    top_k = st.sidebar.slider("¿Top-K resultados?", 1, 15, 5)

    if uploaded_file:
        st.audio(uploaded_file, format="audio/wav" if uploaded_file.name.endswith(".wav") else "audio/mp3")
        st.markdown(f"<small style='color:gray'>Archivo cargado: <b>{uploaded_file.name}</b></small>", unsafe_allow_html=True)

        # Mostrar señal temporal
        # with st.spinner("Cargando visualización de señales del audio..."):
        #     try:
        #         y, sr = librosa.load(uploaded_file, sr=None)
        #         fig, ax = plt.subplots()
        #         librosa.display.waveshow(y, sr=sr, ax=ax)
        #         ax.set_xlabel("Tiempo (s)")
        #         ax.set_ylabel("Amplitud")
        #         st.pyplot(fig)
        #     except Exception as e:
        #         st.warning(f"No se pudo mostrar la forma de onda: {e}")

    if st.button("Buscar canciones similares"):
        if uploaded_file:
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            with st.spinner("Buscando canciones similares..."):

                try:
                    res = requests.post(f"{API_URL}/search_audio", files=files, params={"k": top_k})
                    if res.status_code == 200:
                        data = res.json()

                        # Mostrar resultados por método
                        if "knn_secuencial" in data and "knn_invertido" in data:
                            st.markdown("### 🔁 Resultados con KNN Secuencial")
                            st.dataframe(pd.DataFrame(data["knn_secuencial"]))

                            st.markdown(
                                f"<small style='color:gray'>⏱️ Tiempo de ejecución: {data['tiempos']['secuencial']} ms</small>",
                                unsafe_allow_html=True)

                            st.markdown("### ⚡ Resultados con KNN con Índice Invertido")
                            st.dataframe(pd.DataFrame(data["knn_invertido"]))

                            st.markdown(
                                f"<small style='color:gray'>⏱️ Tiempo de ejecución: {data['tiempos']['invertido']} ms</small>",
                                unsafe_allow_html=True)

                        else:
                            st.warning("No se encontraron resultados válidos.")
                    else:
                        st.error("Error en la búsqueda.")
                except Exception as e:
                    st.error(f"Error de conexión: {e}")
        else:
            st.warning("Primero sube un archivo de audio.")


# --------------------------------- GESTION DE TABLAS -------------------------------

elif page == "📁 Gestión de Tablas":
    st.subheader("📁 Crear nueva tabla e insertar datos")

    table_name = st.text_input("Nombre de la nueva tabla (ej: Videos):")
    csv_file = st.file_uploader("Selecciona un archivo CSV", type=["csv"])
    preview_data = {}

    if csv_file:
        st.markdown("### 👁️ Vista previa del archivo")
        try:
            df_preview = pd.read_csv(csv_file)
            st.dataframe(df_preview.head())
        except Exception as e:
            st.error(f"No se pudo leer el CSV: {e}")

        with st.spinner("Analizando columnas..."):
            res = requests.post(f"{API_URL}/preview_csv", files={"file": csv_file.getvalue()})

        if res.status_code == 200:
            preview_data = res.json()
            colnames = [col["name"] for col in preview_data["columns"]]

            st.markdown("### 🧩 Selección de columnas")

            id_column = st.selectbox(
                "Columna identificadora (opcional)",
                [""] + colnames,
                index=1 if preview_data["suggested_id_columns"] else 0
            )
            id_column = id_column if id_column != "" else None

            text_column = st.selectbox(
                "Columna de texto a indexar (obligatoria)",
                colnames,
                index=colnames.index(preview_data["suggested_text_columns"][0]) if preview_data["suggested_text_columns"] else 0
            )

            if st.button("Crear tabla e insertar datos"):
                if not table_name:
                    st.warning("Debes ingresar un nombre para la tabla.")
                elif not text_column:
                    st.warning("Debes seleccionar una columna de texto.")
                else:
                    with st.spinner("Procesando..."):
                        # Crear tabla + Insertar CSV
                        files = {"file": (csv_file.name, csv_file.getvalue(), "text/csv")}
                        data = {
                            "table": table_name,
                            "text_column": text_column
                        }
                        if id_column:
                            data["id_column"] = id_column

                        res = requests.post(f"{API_URL}/insert_csv", files=files, data=data)

                    if res.status_code == 200:
                        msg = res.json()["message"]
                        st.success(f"✅ {msg}")
                    else:
                        st.error(f"❌ Error: {res.text}")
        else:
            st.error(f"Error al analizar columnas del CSV: {res.text}")
