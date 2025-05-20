import base64
from io import BytesIO  # ¡Aquí está la importación que faltaba!
def audio_to_base64(audio_path):
    with open(audio_path, "rb") as audio_file:
        encoded_audio = base64.b64encode(audio_file.read()).decode("utf-8")
    return encoded_audio

def base64_to_audio(base64_string):
    audio_bytes = base64.b64decode(base64_string)
    return BytesIO(audio_bytes) 