import base64
from io import BytesIO 
from pydub import AudioSegment

def audio_to_base64(audio_path):
    with open(audio_path, "rb") as audio_file:
        encoded_audio = base64.b64encode(audio_file.read()).decode("utf-8")
    return encoded_audio

def base64_to_audio(base64_string):
    audio_bytes = base64.b64decode(base64_string)
    audio_increased=increase_volume(BytesIO(audio_bytes) )
    return audio_increased

def increase_volume(audio_low):
    audio = AudioSegment.from_wav(audio_low)
    strong_audio = audio + 30
    output = BytesIO()
    strong_audio.export(output, format="wav")
    output.seek(0)
    return output.getvalue()