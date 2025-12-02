import json
import math
from collections import Counter
from backend.indexing.preprocessor import preprocess

class SPIMISearcher:
    def __init__(self, index_path="data/Audio/index.json"):
        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.index = data["index"]
            self.doc_norms = data["doc_norms"]

    def search(self, query, top_k=5):
        query_tokens = preprocess(query)
        tf_query = Counter(query_tokens)
        scores = {}
        query_norm = 0.0

        N = len(self.doc_norms)  # nº de docs, para IDF de la query

        for term, tf in tf_query.items():
            postings = self.index.get(term, [])
            df = len(postings)
            if df == 0:
                continue

            idf = math.log(N / (1 + df))   # misma fórmula que en el indexador
            wq = tf * idf
            query_norm += wq ** 2

            for doc_id, w_td in postings:
                doc_id = str(doc_id)
                scores[doc_id] = scores.get(doc_id, 0.0) + w_td * wq

        query_norm = math.sqrt(query_norm) if query_norm > 0 else 1e-6

        for doc_id in list(scores.keys()):
            doc_norm = self.doc_norms.get(str(doc_id), 1e-6)
            scores[doc_id] /= (query_norm * doc_norm)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
