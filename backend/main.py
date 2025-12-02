from fastapi import FastAPI, Query, UploadFile, File, HTTPException,Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from backend.indexing.search import SPIMISearcher
from backend.indexing.spimi import SPIMIIndexer
from backend.audio_processing import transcribe_audio
from backend.ai_query_parser import parse_sql_query
from backend.audio_indexer import AudioIndexer
from backend.models import SearchResponse, SQLQuery
from backend.utils import ensure_identifier_column, detect_text_column, get_audio_files
from typing import List
import pandas as pd
import csv
import os
from datetime import datetime
import time
import uuid
import json
import pydub
from pydub import AudioSegment

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/search", response_model=List[SearchResponse])
def search(q: str, table: str = Query(...), k: int = 5):
    index_path = f"data/{table}/index.json"
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail=f"Índice para tabla '{table}' no encontrado.")
    
    searcher = SPIMISearcher(index_path=index_path)
    results = searcher.search(q, top_k=k)
    return [{"doc_id": doc_id, "score": score} for doc_id, score in results]


# @app.post("/search_audio")
# async def search_from_audio(file: UploadFile = File(...), k: int = 5):
#     AUDIO_INDEX_PATH = "multimedia/audio_index.pkl" # recuperacion de diccionario acustico
#     LOG_CSV_PATH = "analisis/logs_multimedia.csv"

#     if not os.path.exists(AUDIO_INDEX_PATH):
#         return JSONResponse(status_code=404, content={"error": "Índice acústico no encontrado."})
    
#     # Guardar temporalmente el audio subido
#     os.makedirs("temp_audio", exist_ok=True)
#     audio_path = f"temp_audio/{uuid.uuid4()}.mp3"
#     with open(audio_path, "wb") as f:
#         f.write(await file.read())

#     # Cargar índice
#     indexer = AudioIndexer()
#     indexer.load(AUDIO_INDEX_PATH)

#     results = {}
#     logs = []

#     # KNN Secuencial
#     t0 = time.time()
#     knn_seq = indexer.knn_secuencial(audio_path, k=k)
#     t1 = time.time()
#     seq_time_ms = round((t1 - t0) * 1000, 2)
#     results["knn_secuencial"] = [{"doc_id": doc, "score": score} for doc, score in knn_seq]
#     logs.append({"metodo": "KNN-secuencial", "tiempo_respuesta": seq_time_ms})

#     # KNN Invertido
#     t2 = time.time()
#     knn_inv = indexer.knn_invertido(audio_path, k=k)
#     t3 = time.time()
#     inv_time_ms = round((t3 - t2) * 1000, 2)
#     results["knn_indexado"] = [{"doc_id": doc, "score": score} for doc, score in knn_inv]
#     logs.append({"metodo": "KNN-Indexado", "tiempo_respuesta": inv_time_ms})

#     # Guardar logs
#     os.makedirs("analisis", exist_ok=True)
#     log_df = pd.DataFrame(logs)
#     if os.path.exists(LOG_CSV_PATH):
#         log_df.to_csv(LOG_CSV_PATH, mode="a", header=False, index=False)
#     else:
#         log_df.to_csv(LOG_CSV_PATH, index=False)

#     # Eliminar archivo temporal
#     os.remove(audio_path)

#     return {
#         "knn_secuencial": results["knn_secuencial"],
#         "tiempo_secuencial_ms": seq_time_ms,
#         "knn_indexado": results["knn_indexado"],
#         "tiempo_indexado_ms": inv_time_ms
#     }

@app.post("/search_audio")
async def search_from_audio(file: UploadFile = File(...), k: int = 5):
    print("🔧 Endpoint /search_audio activado.")
    os.makedirs("temp_audio", exist_ok=True)

    # Generar nombre temporal
    temp_filename = f"temp_audio/{uuid.uuid4()}.wav"

    try:
        contents = await file.read()
        print(f"📥 Archivo recibido: {file.filename}, tamaño: {len(contents)} bytes")

        with open(temp_filename, "wb") as f:
            f.write(contents)
        print(f"✅ Archivo guardado como {temp_filename}")

        if temp_filename.endswith(".mp3"):
            print("⚠️ Archivo es mp3, convirtiendo a WAV...")
            audio = AudioSegment.from_file(temp_filename, format="mp3")
            temp_wav = temp_filename.replace(".mp3", ".wav")
            audio.export(temp_wav, format="wav")
            os.remove(temp_filename)
            temp_filename = temp_wav
            print(f"✅ Conversión exitosa: {temp_filename}")

        if not os.path.exists(temp_filename):
            print("❌ Error: archivo no existe después de guardar")
            return JSONResponse(status_code=500, content={"error": f"Archivo no guardado: {temp_filename}"})
        elif os.stat(temp_filename).st_size == 0:
            print("❌ Error: archivo vacío")
            return JSONResponse(status_code=500, content={"error": f"Archivo vacío: {temp_filename}"})

        print("📦 Cargando índice acústico...")
        index_path = "multimedia/audio_index.pkl"
        if not os.path.exists(index_path):
            print("❌ No se encontró el índice acústico.")
            return JSONResponse(status_code=500, content={"error": "No se encontró el índice acústico."})

        indexer = AudioIndexer()
        indexer.load(index_path)
        print("✅ Índice cargado con éxito.")

        print("🔁 Ejecutando KNN secuencial...")
        t0 = time.time()
        knn_seq = indexer.knn_secuencial(temp_filename, k=k)
        t_seq = round((time.time() - t0) * 1000, 2)
        print(f"✅ KNN secuencial listo en {t_seq} ms")

        print("⚡ Ejecutando KNN con índice invertido...")
        t0 = time.time()
        knn_inv = indexer.knn_invertido(temp_filename, k=k)
        t_inv = round((time.time() - t0) * 1000, 2)
        print(f"✅ KNN invertido listo en {t_inv} ms")

        print("📝 Guardando log en CSV...")
        os.makedirs("analisis", exist_ok=True)
        log_path = "analisis/logs_multimedia.csv"
        df_log = pd.DataFrame([
            {"metodo": "KNN-secuencial", "tiempo_respuesta": t_seq},
            {"metodo": "KNN-Indexado", "tiempo_respuesta": t_inv}
        ])
        if os.path.exists(log_path):
            df_log.to_csv(log_path, mode="a", header=False, index=False)
        else:
            df_log.to_csv(log_path, index=False)
        print("✅ Log guardado.")

        return {
            "knn_secuencial": [{"doc_id": d, "score": s} for d, s in knn_seq],
            "knn_invertido": [{"doc_id": d, "score": s} for d, s in knn_inv],
            "tiempos": {"secuencial": t_seq, "invertido": t_inv}
        }

    except Exception as e:
        print(f"💥 Excepción atrapada: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        try:
            os.remove(temp_filename)
            print(f"🧹 Archivo temporal eliminado: {temp_filename}")
        except:
            print(f"⚠️ No se pudo eliminar: {temp_filename}")

@app.post("/search_sql")
def search_from_sql(payload: SQLQuery):
    table_name, query_text, k, selected_fields = parse_sql_query(payload.query)
    print("TABLA: ", table_name)
    print("QUERY: ", query_text)
    index_path = f"data/{table_name}/index.json"
    metadata_path = f"data/{table_name}/metadata.csv"

    if not query_text or not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise HTTPException(status_code=400, detail="Consulta inválida o recursos no encontrados.")


    start_time = time.time()

    # Buscar
    searcher = SPIMISearcher(index_path=index_path)
    results = searcher.search(query_text, top_k=k)

    # Cargar metadata
    df = pd.read_csv(metadata_path)

    id_column = None
    if "track_id" in df.columns:
        id_column = "track_id"
    if "id" in df.columns:
        id_column = "id"
    else:
        df["id"] = df.index 
        id_column = "id"
        df.to_csv(metadata_path, index=False)

    df.set_index(id_column, inplace=True)
    df.index = df.index.map(str)
    
    output = []
    for doc_id, score in results:
        if doc_id in df.index:
            row = df.loc[doc_id]
            # enriched = {f: row.get(f, "") for f in selected_fields}
            # enriched["score"] = score
            # output.append(enriched)
            
            enriched = {f: row.get(f, "") for f in selected_fields}
            enriched = {k: (v.item() if hasattr(v, "item") else v) for k, v in enriched.items()} # convertir todos los valores a tipos nativos
            enriched["score"] = float(score)  # asegurar tipo nativo
            output.append(enriched)



    print("Resultados crudos del índice:", results)
    print("Índice del DataFrame:", df.index.tolist()[:10])
    print("Tipos de doc_id en results:", [type(r[0]) for r in results])
    print("Tipos de df.index:", type(df.index[0]))

    end_time = time.time()
    elapsed_ms = round((end_time - start_time) * 1000, 2)

    # Guardar log
    log_path = "analisis/log.csv"
    log_row = {
        "tabla": table_name,
        "tiempo_respuesta": elapsed_ms,
        "query_text": query_text,
        "top_k": k,
        "fecha": datetime.now().isoformat(timespec="seconds")
    }

    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=log_row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_row)


    return output


@app.post("/insert_csv")
def insert_csv(
    table: str = Form(...),
    file: UploadFile = File(...),
    id_column: str = Form(None),
    text_column: str = Form(None)
):

    os.makedirs(f"data/{table}", exist_ok=True)
    df = pd.read_csv(file.file)

    df, used_id = ensure_identifier_column(df, preferred=id_column)

    try:
        used_text = detect_text_column(df, preferred=text_column)
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"error": "No se encontró una columna textual válida para indexar."}
        )

    metadata_path = f"data/{table}/metadata.csv"
    df.to_csv(metadata_path, index=False)

    documents = df[used_text].astype(str).tolist()
    indexer = SPIMIIndexer(f"data/{table}/index.json")
    indexer.index_documents({i: doc for i, doc in enumerate(documents)}) # {i: doc for i, doc in enumerate(documents)}
    indexer._save_index()

    return {
        "message": f"Tabla '{table}' cargada e indexada exitosamente.",
        "id_column": used_id,
        "text_column": used_text
    }


@app.post("/preview_csv")
def preview_csv(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    column_info = []
    id_candidates = []
    text_candidates = []

    for col in df.columns:
        dtype = str(df[col].dtype)
        column_info.append({"name": col, "type": dtype})

        if dtype in ["object", "string"]:
            text_candidates.append(col)
        if col.lower() in ["id", "track_id", "uuid"]:
            id_candidates.append(col)

    return {
        "columns": column_info,
        "suggested_id_columns": id_candidates,
        "suggested_text_columns": text_candidates[:3]  # sugerir solo las 3 primeras
    }
