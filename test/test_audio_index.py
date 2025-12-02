import os
from backend.audio_indexer import AudioIndexer

AUDIO_DIR = "test/audio" # ruta de los audios de prueba
QUERY_AUDIO = "test/query_audio.mp3"  # archivo para buscar similares
INDEX_PATH = "multimedia/audio_index_prueba.pkl"
N_CLUSTERS = 128

# 1. Construcción del diccionario acústico
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

    # Paso 1: entrenar KMeans
    indexer.fit_dictionary(audio_files.values())

    # Paso 2: construir histograma BoAW + TF-IDF
    indexer.build_bow(audio_files)

    # Paso 3: guardar
    indexer.save(INDEX_PATH)

    # Paso 4: cargar y probar KNN
    print("\n🎯 KNN Secuencial:")
    indexer.load(INDEX_PATH)
    knn_seq = indexer.knn_secuencial(QUERY_AUDIO, k=5)
    for doc_id, score in knn_seq:
        print(f"  {doc_id} → score={score:.4f}")

    print("\n🚀 KNN con índice invertido:")
    knn_inv = indexer.knn_invertido(QUERY_AUDIO, k=5)
    for doc_id, score in knn_inv:
        print(f"  {doc_id} → score={score:.4f}")
