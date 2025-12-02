# build_audio_index.py
import os
from backend.audio_indexer import AudioIndexer
import time

AUDIO_DIR = "multimedia/songs"
INDEX_PATH = "multimedia/audio_index.pkl"
N_CLUSTERS = 128

def get_audio_files():
    audio_files = {}
    for fname in os.listdir(AUDIO_DIR):
        if fname.endswith(".wav") or fname.endswith(".mp3"):
            doc_id = fname.split(".")[0]
            audio_files[doc_id] = os.path.join(AUDIO_DIR, fname)
    return audio_files

if __name__ == "__main__":
    print("🔁 Construyendo índice acústico...")

    indexer = AudioIndexer(n_clusters=N_CLUSTERS)
    audio_files = get_audio_files()

    t0 = time.time()
    print("📊 Entrenando diccionario acústico (KMeans)...")
    indexer.fit_dictionary(audio_files.values())
    t_inv = round((time.time() - t0) * 1000, 2)
    print(f"diccionario acustico entrenado en {t_inv} ms")

    print("📚 Construyendo BoAW + TF-IDF...")
    indexer.build_bow(audio_files)

    print(f"💾 Guardando índice en {INDEX_PATH}")
    indexer.save(INDEX_PATH)
    print("✅ Índice construido y guardado correctamente.")
