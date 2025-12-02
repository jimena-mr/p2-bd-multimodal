import librosa
import numpy as np
import os
import pickle
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
from tqdm import tqdm

class AudioIndexer:
    def __init__(self, n_clusters=256):
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.tfidf_transformer = TfidfTransformer()
        self.n_clusters = n_clusters
        self.histograms = {}
        self.doc_ids = []
        self.index_invertido = defaultdict(list)
        self.tfidf_matrix = None

    def extract_mfccs(self, path):
        try:
            y, sr = librosa.load(path, sr=None)
        except Exception as e:
            raise RuntimeError(f"Error al cargar el audio con librosa: {e}")

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512)
        return mfcc.T  # (frames, 13)

    def fit_dictionary(self, audio_paths):
        print("Extrayendo descriptores locales para clustering...")
        all_mfccs = []
        for path in tqdm(audio_paths):
            mfccs = self.extract_mfccs(path)
            all_mfccs.append(mfccs)
        all_mfccs = np.vstack(all_mfccs)
        print(f"Clustering {all_mfccs.shape[0]} vectores MFCC...")
        self.kmeans.fit(all_mfccs)

    def build_bow(self, audio_files_dict):
        """
        audio_files_dict: {doc_id: path_to_audio}
        """
        print("Construyendo BoAW para cada audio...")
        docs = []
        self.doc_ids = list(audio_files_dict.keys())

        for doc_id, path in tqdm(audio_files_dict.items()):
            mfccs = self.extract_mfccs(path)
            labels = self.kmeans.predict(mfccs)
            histogram = np.bincount(labels, minlength=self.n_clusters)
            self.histograms[doc_id] = histogram
            docs.append(histogram)

            # Para índice invertido acústico
            counts = Counter(labels)
            for word_id, freq in counts.items():
                self.index_invertido[word_id].append((doc_id, freq))
        
        print(f"✅ Histogramas BoW generados: {len(self.histograms)} documentos.")
        docs = np.array(docs)
        self.tfidf_matrix = self.tfidf_transformer.fit_transform(docs)
        print(f"✅ TF-IDF entrenado sobre {self.tfidf_matrix.shape[0]} documentos.")

    def save(self, path="multimedia/audio_index.pkl"):
        with open(path, "wb") as f:
            pickle.dump({
                "kmeans": self.kmeans,
                "doc_ids": self.doc_ids,
                "tfidf_matrix": self.tfidf_matrix,
                "index_invertido": self.index_invertido,
                "tfidf_transformer": self.tfidf_transformer 
            }, f)
        print(f"✅ Índice guardado en {path}")

    def load(self, path="multimedia/audio_index.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.kmeans = data["kmeans"]
            self.doc_ids = data["doc_ids"]
            self.tfidf_matrix = data["tfidf_matrix"]
            self.index_invertido = data["index_invertido"]
            self.tfidf_transformer = data["tfidf_transformer"] 
            self.n_clusters = self.kmeans.n_clusters
        
    def knn_secuencial(self, query_path, k=5):
        if not hasattr(self.tfidf_transformer, 'idf_'):
            raise RuntimeError("TfidfTransformer no está entrenado.")

        mfccs = self.extract_mfccs(query_path)
        labels = self.kmeans.predict(mfccs)
        hist = np.bincount(labels, minlength=self.n_clusters).reshape(1, -1)
        hist_tfidf = self.tfidf_transformer.transform(hist)
        sims = cosine_similarity(hist_tfidf, self.tfidf_matrix)[0]
        top_k = np.argsort(sims)[::-1][:k]
        return [(self.doc_ids[i], sims[i]) for i in top_k]

    def knn_invertido(self, query_path, k=5):

        mfccs = self.extract_mfccs(query_path)
        labels = self.kmeans.predict(mfccs)
        query_hist = Counter(labels)
        scores = defaultdict(float)

        for word_id, freq in query_hist.items():
            for doc_id, tf in self.index_invertido.get(word_id, []):
                scores[doc_id] += tf * freq  # TF-TF score simple

        if not scores:
            return []

        scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return scores[:k]


# fit_dictionary(audio_paths) → construye KMeans
# build_bow({doc_id: path}) → genera histograma y TF-IDF
# knn_secuencial(query_path) → búsqueda con similitud coseno
# knn_invertido(query_path) → búsqueda con índice invertido acústico
# save() / load() para reusar modelos