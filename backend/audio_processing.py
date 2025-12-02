import whisper
import tempfile
import os

model = whisper.load_model("base")  # Puedes usar "small" o "medium" si tienes GPU

def transcribe_audio(file) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    try:
        result = model.transcribe(tmp_path, language='es')
        return result["text"]
    finally:
        os.remove(tmp_path)
