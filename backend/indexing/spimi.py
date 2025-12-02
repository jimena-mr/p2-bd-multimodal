# import os
# import math
# import json
# from collections import defaultdict, Counter
# from backend.indexing.preprocessor import preprocess

# class SPIMIIndexer:
#     def __init__(self, output_path="data/Audio/index.json"):
#         self.index = defaultdict(list)
#         self.doc_norms = {}
#         self.output_path = output_path

#     def index_documents(self, documents):
#         N = len(documents)

#         for doc_id, text in documents.items():
#             tokens = preprocess(text)
#             tf = Counter(tokens)
#             norm = 0

#             for term, freq in tf.items():
#                 # IDF usa el DF actual acumulado
#                 df = len(set(post[0] for post in self.index[term])) if term in self.index else 0
#                 idf = math.log(N / (1 + df))
#                 tfidf = freq * idf
#                 self.index[term].append((str(doc_id), tfidf))
#                 norm += tfidf ** 2

#             self.doc_norms[str(doc_id)] = math.sqrt(norm)

#         self._save_index()

#     def _save_index(self):
#         data = {
#             "index": {term: postings for term, postings in self.index.items()},
#             "doc_norms": self.doc_norms
#         }
#         with open(self.output_path, "w", encoding="utf-8") as f:
#             json.dump(data, f)

# backend/indexing/spimi.py  (por ejemplo)

import os
import math
import json
import heapq
from collections import defaultdict, Counter
from backend.indexing.preprocessor import preprocess


class SPIMIIndexer:
    """
    Implementación de Single-Pass In-Memory Indexing (SPIMI)
    con external sort y MergeBlocks.

    Parámetros nuevos (opcionales, no rompen el código actual):
    - block_dir: carpeta para los bloques parciales en disco.
    - max_terms_in_block: nº aprox. de términos distintos por bloque
      (simula 'free memory available' del algoritmo).
    - max_blocks_in_memory: B buffers de RAM usados en MergeBlocks
      (usamos B-1 como nº máximo de bloques de entrada a la vez).
    """

    def __init__(
        self,
        output_path="data/Audio/index.json",
        block_dir="data/Audio/spimi_blocks",
        max_terms_in_block=50_000,
        max_blocks_in_memory=8,
    ):
        self.output_path = output_path
        self.block_dir = block_dir
        os.makedirs(self.block_dir, exist_ok=True)

        # Aproximación a la memoria disponible por bloque
        self.max_terms_in_block = max_terms_in_block

        # B buffers totales en MergeBlocks (B-1 input, 1 output)
        self.max_blocks_in_memory = max_blocks_in_memory

    # ---------- Fase 1: BSBIndexConstruction (recorrer docs y generar bloques) ----------

    def index_documents(self, documents: dict):
        """
        Entrada se mantiene igual:
        - documents: dict {doc_id: texto}
        """
        N = len(documents)  # nº total de documentos (para IDF)

        # 1) Generar bloques usando SPIMI-INVERT
        block_files = self._build_spimi_blocks(documents)

        # 2) MergeBlocks usando B buffers (external sort)
        merged_file = self._merge_blocks(block_files)

        # 3) A partir del bloque final, calcular TF-IDF y normas
        self._compute_tfidf_and_save(merged_file, N)

    # ---------- SPIMI-INVERT: construir un bloque en memoria y escribirlo a disco ----------

    def _build_spimi_blocks(self, documents):
        """
        Implementa el bucle de BSBIndexConstruction + SPIMI-INVERT(token_stream).

        Creamos diccionarios locales (hash) por bloque:
           term -> {doc_id: tf}

        Cuando el nº de términos distintos supera max_terms_in_block,
        se "cierra" el bloque y se escribe a disco.
        """
        block_files = []
        current_dict = defaultdict(lambda: defaultdict(int))
        block_id = 0

        for doc_id, text in documents.items():
            tokens = preprocess(text)
            freqs = Counter(tokens)
            sdoc = str(doc_id)

            # SPIMI-INVERT: acumulamos postings directamente en el hash
            for term, tf in freqs.items():
                current_dict[term][sdoc] += tf

            # Simulación de "free memory available"
            if len(current_dict) >= self.max_terms_in_block:
                path = os.path.join(self.block_dir, f"block_{block_id}.json")
                self._write_block(current_dict, path)
                block_files.append(path)
                block_id += 1
                current_dict.clear()  # vaciamos memoria (nuevo bloque)

        # Último bloque (si quedó algo)
        if current_dict:
            path = os.path.join(self.block_dir, f"block_{block_id}.json")
            self._write_block(current_dict, path)
            block_files.append(path)

        return block_files

    def _write_block(self, dictionary, path):
        """
        WriteBlockToDisk(sorted_terms, dictionary, output_file)

        - Ordena los términos del diccionario local.
        - Convierte term -> {doc_id: tf} en term -> [(doc_id, tf), ...]
        - Guarda el bloque en JSON.
        """
        sorted_terms = sorted(dictionary.keys())
        block_index = {}

        for term in sorted_terms:
            postings_dict = dictionary[term]               # {doc_id: tf}
            postings_list = sorted(postings_dict.items(),  # [(doc_id, tf)]
                                   key=lambda x: x[0])
            block_index[term] = postings_list

        with open(path, "w", encoding="utf-8") as f:
            json.dump(block_index, f)

    # ---------- Fase 2: MergeBlocks usando B buffers (external k-way merge) ----------

    def _merge_blocks(self, block_files):
        """
        MergeBlocks(f1,...,fn; f_merged)

        Hacemos un merge multi-vía limitado por B buffers:
        - fan_in = B-1 bloques de entrada (cada uno es un 'input buffer').
        - 1 bloque de salida (output buffer).
        - Si hay más de B-1 bloques, se hacen varias rondas hasta quedar uno.

        Devuelve la ruta del bloque final (todavía con TF crudo, sin TF-IDF).
        """
        if not block_files:
            raise ValueError("No hay bloques SPIMI para fusionar")

        if len(block_files) == 1:
            return block_files[0]

        fan_in = max(2, self.max_blocks_in_memory - 1)  # B-1
        round_idx = 0
        files = block_files

        while len(files) > 1:
            new_files = []
            for i in range(0, len(files), fan_in):
                group = files[i : i + fan_in]
                if len(group) == 1:
                    # Ese bloque pasa tal cual a la siguiente ronda
                    new_files.append(group[0])
                    continue

                merged_path = os.path.join(
                    self.block_dir, f"merged_r{round_idx}_{len(new_files)}.json"
                )
                self._merge_group(group, merged_path)
                new_files.append(merged_path)

                # Limpiamos los bloques ya fusionados (opcional pero recomendable)
                for p in group:
                    try:
                        os.remove(p)
                    except OSError:
                        pass

            files = new_files
            round_idx += 1

        # Al final queda un solo bloque
        return files[0]

    def _merge_group(self, group_files, output_path):
        # cargar bloques como mapas
        blocks = []
        for path in group_files:
            with open(path, "r", encoding="utf-8") as f:
                blocks.append(json.load(f))

        # merge real SPIMI: term → {doc_id: tf}
        merged = {}

        for block in blocks:
            for term, postings in block.items():
                if term not in merged:
                    merged[term] = postings.copy()
                else:
                    for doc_id, tf in postings.items():
                        merged[term][doc_id] = merged[term].get(doc_id, 0) + tf

        # ordenar términos
        merged_sorted = dict(sorted(merged.items(), key=lambda x: x[0]))

        # guardar
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged_sorted, f)

    # ---------- Fase 3: calcular TF-IDF y normas a partir del bloque final ----------

    def _compute_tfidf_and_save(self, merged_file, N):
        """
        A partir del bloque final (term -> [(doc_id, tf)]), calculamos:

        - DF(term) = nº de documentos en postings.
        - IDF(term) = log(N / (1 + DF(term)))   [mismo esquema que usabas].
        - w_td = tf * idf
        - doc_norms[d] = ||d||2 = sqrt(sum_t w_td^2)

        Luego guardamos en output_path con el mismo formato que tu
        implementación original:
            {
              "index": { term: [(doc_id, w_td), ...], ... },
              "doc_norms": { doc_id: ||d||, ... }
            }
        """
        with open(merged_file, "r", encoding="utf-8") as f:
            raw_index = json.load(f)

        index_weighted = {}
        doc_norms = {}

        for term, postings in raw_index.items():
            df = len(postings)
            idf = math.log(N / (1 + df))  # misma fórmula que tenías antes

            new_postings = []
            for doc_id, tf in postings:
                w = tf * idf
                new_postings.append((doc_id, w))
                doc_norms[doc_id] = doc_norms.get(doc_id, 0.0) + w * w

            index_weighted[term] = new_postings

        # Norma Euclídea de cada documento
        for doc_id in doc_norms:
            doc_norms[doc_id] = math.sqrt(doc_norms[doc_id])

        # Guardamos índice final (único archivo)
        data = {"index": index_weighted, "doc_norms": doc_norms}
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
