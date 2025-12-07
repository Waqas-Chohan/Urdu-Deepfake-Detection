import streamlit as st
import numpy as np
import librosa
import joblib
import json
from tensorflow import keras
from keras.models import load_model
import pandas as pd
import plotly.graph_objects as go
from io import BytesIO
import os
import matplotlib.pyplot as plt
import librosa.display

# Page configuration
st.set_page_config(
    page_title="Urdu Deepfake Audio Detection",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Global Theme & Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background: radial-gradient(circle at top left, #1a1c24, #0e1117);
    }
    
    /* Headers */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4ECDC4, #FF6B6B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .sub-header {
        font-size: 1.25rem;
        color: #A0AEC0;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #F7FAFC;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #4ECDC4;
        padding-left: 1rem;
    }
    
    /* Glassmorphism Containers */
    .glass-container {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 1rem;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }
    
    .glass-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.2);
        border-color: rgba(78, 205, 196, 0.3);
    }
    
    /* Result Cards */
    .result-card {
        padding: 2.5rem;
        border-radius: 1.5rem;
        text-align: center;
        animation: fadeIn 0.8s cubic-bezier(0.2, 0.8, 0.2, 1);
        margin: 2rem 0;
        backdrop-filter: blur(12px);
    }
    
    .bonafide-card {
        background: linear-gradient(135deg, rgba(40, 167, 69, 0.1), rgba(40, 167, 69, 0.2));
        border: 1px solid #28a745;
        box-shadow: 0 0 30px rgba(40, 167, 69, 0.2);
    }
    
    .spoofed-card {
        background: linear-gradient(135deg, rgba(220, 53, 69, 0.1), rgba(220, 53, 69, 0.2));
        border: 1px solid #dc3545;
        box-shadow: 0 0 30px rgba(220, 53, 69, 0.2);
    }
    
    .result-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .result-text {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: 1px;
    }
    
    .confidence-text {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.8);
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* File Uploader Customization */
    [data-testid='stFileUploader'] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 1rem;
        padding: 2rem;
        border: 2px dashed rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    [data-testid='stFileUploader']:hover {
        border-color: #4ECDC4;
        background-color: rgba(78, 205, 196, 0.05);
    }
    
    /* Custom Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0E1117;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #4ECDC4, #556270);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        width: 100%;
    }
    
    .stButton > button:hover {
        box-shadow: 0 4px 12px rgba(78, 205, 196, 0.4);
        transform: translateY(-1px);
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-enter {
        animation: fadeIn 0.6s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# AudioPreprocessor class
class AudioPreprocessor:
    """Preprocessing pipeline for audio feature extraction"""

    def __init__(self, sr=16000, duration=4.0, n_mfcc=13, n_mel=128, n_chroma=12):
        self.sr = sr
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.n_mel = n_mel
        self.n_chroma = n_chroma
        self.max_samples = int(sr * duration)

    def pad_or_truncate(self, audio):
        if len(audio) < self.max_samples:
            audio = np.pad(audio, (0, self.max_samples - len(audio)), mode='constant')
        else:
            audio = audio[:self.max_samples]
        return audio

    def extract_mfcc(self, audio):
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=self.n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        return np.concatenate([mfcc_mean, mfcc_std])

    def extract_mel_spectrogram(self, audio):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_mels=self.n_mel)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_mean = np.mean(mel_spec_db, axis=1)
        mel_std = np.std(mel_spec_db, axis=1)
        return np.concatenate([mel_mean, mel_std])

    def extract_chroma(self, audio):
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sr, n_chroma=self.n_chroma)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        return np.concatenate([chroma_mean, chroma_std])

    def extract_zero_crossing_rate(self, audio):
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        return np.array([np.mean(zcr), np.std(zcr)])

    def extract_spectral_centroid(self, audio):
        spec_cent = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
        return np.array([np.mean(spec_cent), np.std(spec_cent)])

    def extract_features(self, audio):
        audio = self.pad_or_truncate(audio)
        mfcc_feat = self.extract_mfcc(audio)
        mel_feat = self.extract_mel_spectrogram(audio)
        chroma_feat = self.extract_chroma(audio)
        zcr_feat = self.extract_zero_crossing_rate(audio)
        spec_cent_feat = self.extract_spectral_centroid(audio)
        
        features = np.concatenate([
            mfcc_feat,
            mel_feat,
            chroma_feat,
            zcr_feat,
            spec_cent_feat
        ])
        return features

@st.cache_resource
def load_models_and_data():
    """Load all models and configuration files"""
    try:
        # Load models
        svm_model = joblib.load('svm_model.pkl')
        lr_model = joblib.load('logistic_regression_model.pkl')
        perceptron_model = joblib.load('perceptron_model.pkl')
        dnn_model = load_model('dnn_model.keras')
        
        # Load scaler
        scaler = joblib.load('scaler.pkl')
        
        # Load configurations
        with open('preprocessor_config.json', 'r') as f:
            preprocessor_config = json.load(f)
        
        with open('label_mapping.json', 'r') as f:
            label_mapping = json.load(f)
        
        # Initialize preprocessor (exclude max_samples as it's calculated, not a parameter)
        preprocessor_params = {k: v for k, v in preprocessor_config.items() if k != 'max_samples'}
        preprocessor = AudioPreprocessor(**preprocessor_params)
        
        models = {
            'SVM': svm_model,
            'Logistic Regression': lr_model,
            'Single-Layer Perceptron': perceptron_model,
            'Deep Neural Network': dnn_model
        }
        
        return models, scaler, preprocessor, label_mapping
    
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Please ensure all model files are in the same directory as this app.")
        return None, None, None, None

def predict_audio(audio_data, sr, model_name, models, scaler, preprocessor):
    """Make prediction on audio data"""
    try:
        # Resample if needed
        if sr != preprocessor.sr:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=preprocessor.sr)
        
        # Extract features
        features = preprocessor.extract_features(audio_data)
        features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Get model
        model = models[model_name]
        
        # Make prediction
        if model_name == 'Deep Neural Network':
            prediction_proba = model.predict(features_scaled, verbose=0)[0][0]
            prediction = 1 if prediction_proba >= 0.5 else 0
        else:
            prediction = model.predict(features_scaled)[0]
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(features_scaled)[0][1]
            else:
                # For perceptron, use decision function
                decision = model.decision_function(features_scaled)[0]
                prediction_proba = 1 / (1 + np.exp(-decision))  # Sigmoid
        
        return prediction, prediction_proba
    
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, None

def plot_waveform(audio_data, sr):
    """Plot audio waveform as static image"""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    time = np.arange(0, len(audio_data)) / sr
    ax.plot(time, audio_data, color='#4ECDC4', linewidth=0.5)
    ax.set_title('Audio Waveform', fontsize=12, fontweight='bold', color='white')
    ax.set_xlabel('Time (seconds)', fontsize=10, color='gray')
    ax.set_ylabel('Amplitude', fontsize=10, color='gray')
    ax.tick_params(colors='gray')
    ax.grid(True, alpha=0.1, color='white')
    for spine in ax.spines.values():
        spine.set_color('gray')
        
    plt.tight_layout()
    return fig

def plot_spectrogram(audio_data, sr):
    """Plot mel spectrogram as static image"""
    plt.style.use('dark_background')
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    img = librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', 
                                    sr=sr, cmap='magma', ax=ax)
    ax.set_title('Mel Spectrogram', fontsize=12, fontweight='bold', color='white')
    ax.set_xlabel('Time (seconds)', fontsize=10, color='gray')
    ax.set_ylabel('Mel Frequency (Hz)', fontsize=10, color='gray')
    ax.tick_params(colors='gray')
    
    # Custom colorbar
    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.ax.yaxis.set_tick_params(color='gray')
    cbar.outline.set_edgecolor('gray')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='gray')
    
    plt.tight_layout()
    return fig

# Main app
def main():
    # Sidebar - specific container for controls
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        st.markdown("Select detection model and settings")
        
        # Load models quietly first to get names
        models_data, _, _, _ = load_models_and_data()
        
        if models_data:
            selected_model = st.selectbox(
                "Detection Model",
                options=list(models_data.keys()),
                index=0
            )
        else:
            selected_model = None
            
        st.markdown("---")
        st.markdown("""
        <div style='background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 0.5rem; font-size: 0.8rem; color: #888;'>
            <strong>About:</strong><br>
            This system uses advanced machine learning to detect synthetic audio generation (Deepfakes) in Urdu language samples.
        </div>
        """, unsafe_allow_html=True)

    # Main Area
    # Header
    st.markdown('<div class="main-header">Urdu Deepfake Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced Audio Authenticity Verification System</div>', unsafe_allow_html=True)
    
    # Load models full
    models, scaler, preprocessor, label_mapping = load_models_and_data()
    
    if models is None:
        st.stop()
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    
    # Upload Section in a beautified container
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📁 Input Audio</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload a WAV, MP3, or M4A file (Max 10MB)",
        type=['wav', 'mp3', 'm4a', 'ogg'],
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Check if this is a new file
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.current_file != file_id:
            st.session_state.current_file = file_id
            st.session_state.analysis_results = None
        
        # Temporary file handling
        try:
            temp_file_path = f"temp_audio_{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load audio
            audio_data, sr = librosa.load(temp_file_path, sr=None)
            
            # Clean up
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            
            # 2-Column Layout for Player and Info
            col_player, col_info = st.columns([1, 1])
            
            with col_player:
                st.markdown('<div style="margin-top: 5px;"></div>', unsafe_allow_html=True)
                st.audio(uploaded_file, format='audio/wav')
                
            with col_info:
                duration = len(audio_data) / sr
                st.markdown(f"""
                <div style="display: flex; gap: 1rem; align-items: center; background: rgba(255,255,255,0.05); padding: 0.5rem 1rem; border-radius: 0.5rem; border: 1px solid rgba(255,255,255,0.1);">
                    <span>⏱️ <b>{duration:.1f}s</b></span>
                    <span>📊 <b>{sr} Hz</b></span>
                </div>
                """, unsafe_allow_html=True)
            
            # Analyze Button
            if st.button("🔍 START ANALYSIS", type="primary", use_container_width=True):
                with st.spinner(f'Processing acoustics with {selected_model}...'):
                    prediction, confidence = predict_audio(
                        audio_data, sr, selected_model, models, scaler, preprocessor
                    )
                    
                    if prediction is not None:
                        st.session_state.analysis_results = {
                            'prediction': prediction,
                            'confidence': confidence,
                            'model_name': selected_model,
                            'audio_data': audio_data,
                            'sr': sr
                        }
            
            # Display Result
            if st.session_state.analysis_results is not None:
                results = st.session_state.analysis_results
                prediction = results['prediction']
                confidence = results['confidence']
                
                # Determine class and style
                if prediction == 0:
                    result_type = "bonafide"
                    icon = "🛡️"
                    label = "BONAFIDE (REAL)"
                    actual_confidence = 1 - confidence
                    color = "#28a745"
                else:
                    result_type = "spoofed"
                    icon = "⚠️"
                    label = "SPOOFED (DEEPFAKE)"
                    actual_confidence = confidence
                    color = "#dc3545"
                
                # Result Card
                st.markdown(f"""
                <div class="result-card {result_type}-card">
                    <span class="result-icon">{icon}</span>
                    <div class="result-text">{label}</div>
                    <div class="confidence-text">Confidence: {actual_confidence:.2%}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Detailed Analysis Section
                st.markdown('<div class="glass-container animate-enter">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">📊 Acoustic Analysis</div>', unsafe_allow_html=True)
                
                # Gauge Chart
                col_gauge, col_viz = st.columns([1, 2])
                
                with col_gauge:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=actual_confidence * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Confidence Score", 'font': {'color': 'white', 'size': 16}},
                        number={'font': {'color': 'white'}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickcolor': "white", 'tickwidth': 2},
                            'bar': {'color': color},
                            'bgcolor': "rgba(0,0,0,0)",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 100], 'color': "rgba(255,255,255,0.1)"}
                            ],
                        }
                    ))
                    fig_gauge.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=250,
                        margin=dict(l=20, r=20, t=30, b=20)
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                with col_viz:
                    tab1, tab2 = st.tabs(["Waveform", "Spectrogram"])
                    with tab1:
                        fig_wave = plot_waveform(results['audio_data'], results['sr'])
                        st.pyplot(fig_wave, use_container_width=True)
                        plt.close(fig_wave)
                    with tab2:
                        fig_spec = plot_spectrogram(results['audio_data'], results['sr'])
                        st.pyplot(fig_spec, use_container_width=True)
                        plt.close(fig_spec)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            
    else:
        # Welcome / Empty State
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; color: #888;">
            <h3>👋 Welcome to the Authenticity Lab</h3>
            <p>Upload an audio file to begin your forensic analysis.</p>
            <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 2rem;">
                <div>
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🎙️</div>
                    <small>High Quality</small>
                </div>
                <div>
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">✨</div>
                    <small>Instant Results</small>
                </div>
                <div>
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🔒</div>
                    <small>Secure Process</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
