import pandas as pd
import os

AUDIO_DIR = "multimedia/songs"

def get_audio_files():
    audio_files = {}
    for fname in os.listdir(AUDIO_DIR):
        if fname.endswith(".wav") or fname.endswith(".mp3"):
            doc_id = fname.split(".")[0]
            audio_files[doc_id] = os.path.join(AUDIO_DIR, fname)
    return audio_files
    
def ensure_identifier_column(df: pd.DataFrame, preferred: str = None) -> tuple[pd.DataFrame, str]:
    if preferred and preferred in df.columns:
        return df, preferred
    for col in ["track_id", "id", "uuid"]:
        if col in df.columns:
            return df, col
    df = df.reset_index(drop=True)
    df["id"] = df.index
    return df, "id"

def detect_text_column(df: pd.DataFrame, preferred: str = None) -> str:
    if preferred and preferred in df.columns and df[preferred].dtype == object:
        return preferred
    for col in df.columns:
        if df[col].dtype == object and df[col].notnull().any():
            return col
    raise ValueError("No se encontró una columna textual válida para indexar.")
