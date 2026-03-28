# src/audio_utils.py

import speech_recognition as sr
import tempfile
from pydub import AudioSegment
import math

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()

    audio = AudioSegment.from_wav(audio_path)
    duration_ms = len(audio)
    chunk_length_ms = 60000
    chunks = math.ceil(duration_ms / chunk_length_ms)

    full_text = ""

    for i in range(chunks):
        start = i * chunk_length_ms
        end = min((i + 1) * chunk_length_ms, duration_ms)
        chunk = audio[start:end]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_chunk:
            chunk.export(tmp_chunk.name, format="wav")
            tmp_chunk_path = tmp_chunk.name

        with sr.AudioFile(tmp_chunk_path) as source:
            audio_listened = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_listened)
                full_text += " " + text
            except:
                pass

    return full_text.strip()