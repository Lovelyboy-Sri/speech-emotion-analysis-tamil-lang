import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU

import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import json
import random
from gtts import gTTS
import tempfile
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av

# ---------------- Lazy TensorFlow import ----------------
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------------- Load Model ----------------
# Uncomment and update path to your model
model = load_model("/workspaces/speech-emotion-analysis-tamil-lang/emotionclassifier.h5")

# ---------------- Load Responses ----------------
with open("/workspaces/speech-emotion-analysis-tamil-lang/fastapi/response.json", "r") as f:
    emotion_responses = json.load(f)

# ---------------- Emotions ----------------
emotions = ['angry', 'fear', 'happy', 'neutral', 'sad']

# ---------------- Feature Extraction ----------------
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    x = np.expand_dims(mfccs_mean, axis=(0, -1))  # shape (1, 13, 1)
    return x

# ---------------- Prediction ----------------
def predict_emotion(file_path):
    x = extract_features(file_path)
    if 'model' in globals():
        prediction = model.predict(x)
        predicted_emotion = emotions[prediction.argmax(axis=1)[0]]
        prediction_percentages = {
            emotions[i]: round(float(prediction[0][i] * 100),2) for i in range(len(emotions))
        }
        
    else:
        # Placeholder if model not loaded
        predicted_emotion = random.choice(emotions)
        prediction_percentages = {e: random.randint(10, 30) for e in emotions}
    return predicted_emotion, prediction_percentages

# ---------------- Suggestion + TTS ----------------
def get_dynamic_suggestion(emotion):
    suggestions = emotion_responses.get(emotion, ["No suggestions available."])
    suggestion = random.choice(suggestions)
    tts = gTTS(suggestion, lang="en")
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp_file.name)
    return suggestion, tmp_file.name

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="🎤 Emotion Detection", layout="centered")
st.title("🎤 Emotion Detection (Tanglish Suggestions + Voice)")
st.write("Upload a file or record live voice for emotion detection.")

# ---------------- File Upload ----------------
st.subheader("📂 Upload an Audio File")
uploaded_file = st.file_uploader("Upload a .wav or .mp3 file", type=["wav", "mp3"])
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    predicted_emotion, percentages = predict_emotion(tmp_path)
    st.success(f"Predicted Emotion: **{predicted_emotion}**")
    st.json(percentages)
    st.bar_chart(percentages)

    suggestion, voice_file = get_dynamic_suggestion(predicted_emotion)
    st.subheader("💡 Suggested Response (Tanglish):")
    st.write(suggestion)
    st.audio(voice_file, format="audio/mp3")

# ---------------- Live Recording ----------------
st.subheader("🎙️ Record Your Voice (Press Start/Stop)")

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recorded_frames = []

    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        self.recorded_frames.append(audio)
        return frame

webrtc_ctx = webrtc_streamer(
    key="speech-emotion",
    mode=WebRtcMode.SENDRECV,
    audio_receiver_size=1024,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
    audio_processor_factory=AudioProcessor,
)

if webrtc_ctx and webrtc_ctx.state.playing:
    if st.button("🔴 Stop & Analyze"):
        audio_frames = webrtc_ctx.audio_processor.recorded_frames
        if audio_frames:
            audio_np = np.concatenate(audio_frames, axis=0).astype(np.float32)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                sf.write(tmp_file.name, audio_np, 22050)
                tmp_path = tmp_file.name

            st.audio(tmp_path, format="audio/wav")
            predicted_emotion, percentages = predict_emotion(tmp_path)
            st.success(f"Predicted Emotion: **{predicted_emotion}**")
            st.json(percentages)
            st.bar_chart(percentages)

            suggestion, voice_file = get_dynamic_suggestion(predicted_emotion)
            st.subheader("💡 Suggested Response (Tanglish):")
            st.write(suggestion)
            st.audio(voice_file, format="audio/mp3")
        else:
            st.warning("No audio recorded yet.")
