import numpy as np
import tensorflow as tf
from pydub import AudioSegment
import librosa
import joblib
import cv2
import psutil
import tracemalloc
import os
import time
import assemblyai as aai
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

# Load Models
cnn_model = load_model("./combined_model/models/cnn_six_seconds_ver2.keras")  # Load trained CNN model
nb_model = joblib.load("./combined_model/models/best_model.pkl")# Load trained Naïve Bayes model
vectorizer = joblib.load("./combined_model/models/tfidf_vectorizer.pkl")

load_dotenv()

# Define Weights
cnn_weight = 0.50
nb_weight = 0.88 

# Normalize weights so they sum to 1
total_weight = cnn_weight + nb_weight
cnn_weight /= total_weight
nb_weight /= total_weight

class_labels = {0: "Negative", 1: "Neutral", 2:"Positive"}

# Function to measure memory usage
def get_memory_usage():
    return psutil.Process(os.getpid()).memory_info().rss

# ============================================================== SPEECH PREPROCESS =======================================================
def reduce_noise(y, sr):
    # Estimate noise profile from the first 0.5 seconds
    noise_sample = y[:int(sr * 0.5)]
    reduced_y = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample)
    return reduced_y

def preprocess_audio(audio_file, target_size=(128, 256)):
    y, sr = librosa.load(audio_file, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    mel_spec_normalized = cv2.normalize(mel_spec_db, None, 0, 255, cv2.NORM_MINMAX) # Normalize spectrogram values to 0-255
    mel_spec_resized = cv2.resize(mel_spec_normalized, target_size, interpolation=cv2.INTER_AREA) # Resize to match CNN input

    spectogram = np.array(mel_spec_resized).reshape(-1, 128, 256, 1)
    spectogram = spectogram / 255.0

    return np.expand_dims(spectogram, axis=1) # expanding for time dimension

# ============================================================== SPEECH MODEL =======================================================
def predict_audio(audio_input):
    tracemalloc.start()
    start_mem = get_memory_usage()
    start_time = time.time()

    cnn_probabilities = cnn_model.predict(audio_input)[0]
    predicted_class = np.argmax(cnn_probabilities)  # Get index of highest probability

    end_time = time.time()
    end_mem = get_memory_usage()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    predicted_emotion = class_labels[predicted_class]  # Convert index to emotion label
    print(f"Speech Model Prediction: {predicted_emotion}")

    return cnn_probabilities, end_time - start_time, end_mem - start_mem, peak_mem

# ============================================================== TEXT PREPROCESS =======================================================
def speech_to_text(audio_file):
    aai.settings.api_key = os.getenv("AAI_API_KEY")

    
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_file)

    if transcript.status == aai.TranscriptStatus.error:
        print(f"Transcription failed")
    
    print(f"Transcribed Text: {transcript.text}")
    return [transcript.text]

def preprocess_text(text):
    text = text.lower() # Lowercase text
    text = ''.join([char for char in text if char.isalpha() or char.isspace()]) # Remove punctuation, numbers, etc.
    return text

# ============================================================== TEXT MODEL =======================================================
def predict_text(text):
    tracemalloc.start()
    start_mem = get_memory_usage()
    start_time = time.time()

    processed_data = [preprocess_text(text) for text in text]
    # Convert text to feature vector
    text_vector = vectorizer.transform(processed_data)  # Convert to TF-IDF features

    # Get Naïve Bayes probabilities
    nb_probabilities = nb_model.predict_proba(text_vector)[0]  # (num_classes,)
    
    end_time = time.time()
    end_mem = get_memory_usage()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Prediction using the loaded model
    prediction = nb_model.predict(text_vector)
    print(f"Text Model Prediction: {prediction}")

    return nb_probabilities, end_time - start_time, end_mem - start_mem, peak_mem

# ============================================================== FUSION MODEL =======================================================

def fusion_prediction(audio_input, text):
    # Get probabilities from both models
    cnn_probs, cnn_time, cnn_mem, cnn_peak = predict_audio(audio_input)
    nb_probs, nb_time, nb_mem, nb_peak = predict_text(text)

    final_scores = (cnn_weight * cnn_probs) + (nb_weight * nb_probs) # Compute final weighted score
    final_prediction = np.argmax(final_scores) # Get final prediction (emotion with highest score)

    return (
        final_prediction, final_scores, 
        cnn_time, nb_time,
        cnn_mem, nb_mem,
        cnn_peak, nb_peak
    )

# ============================================================== PREDICTION =======================================================

audio_file = "./combined_model/prediction_test/negative_test_6seconds.wav"  # audio file
text = speech_to_text(audio_file)
audio_input = preprocess_audio(audio_file)

(
    predicted_emotion, confidence_scores, 
    cnn_time, nb_time, 
    cnn_mem, nb_mem, 
    cnn_peak, nb_peak
) = fusion_prediction(audio_input, text)

# ============================================================== RESULTS =======================================================
print(f"Final Predicted Emotion: {predicted_emotion}: {class_labels[predicted_emotion]}")
print("Confidence Scores:", confidence_scores)

print("\n===== TIME COMPLEXITY (Seconds) =====")
print(f"Speech Model: {cnn_time:.4f} sec")
print(f"Text Model: {nb_time:.4f} sec")

print("\n===== SPACE COMPLEXITY (Memory Usage in Bytes) =====")
print(f"Speech Model: {cnn_mem} bytes | Peak: {cnn_peak} bytes")
print(f"Text Model: {nb_mem} bytes | Peak: {nb_peak} bytes")