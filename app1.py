import streamlit as st
import joblib
import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import plotly.graph_objects as go

# Load model
model = joblib.load('beach_safety_model (1).pkl')

# Wind direction mapping
wind_dir_map = {'N': 0, 'NE': 1, 'E': 2, 'SE': 3, 'S': 4, 'SW': 5, 'W': 6, 'NW': 7}

API_KEY = st.secrets["OPENWEATHER_API_KEY"]

def deg_to_compass(deg):
    directions = list(wind_dir_map.keys())
    idx = int((deg + 22.5) / 45.0) % 8
    return directions[idx]

def fetch_weather_data(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"API returned status code {response.status_code}")
    data = response.json()
    try:
        temperature = data['main']['temp']
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed'] * 3.6
        wind_deg = data['wind']['deg']
        wind_dir = deg_to_compass(wind_deg)
        visibility = data.get('visibility', 8000) / 1000
        rainfall = data.get('rain', {}).get('1h', 0)
        wave_height = 1.0
        tide_level = 1.0
        uv_index = 5
        return {
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "wind_direction": wind_dir,
            "wave_height": wave_height,
            "tide_level": tide_level,
            "visibility": visibility,
            "rainfall": rainfall,
            "uv_index": uv_index
        }
    except KeyError as e:
        raise ValueError(f"Missing key in API response: {e}")

def predict_beach_safety(data):
    wind_encoded = wind_dir_map.get(data['wind_direction'], 0)
    features = np.array([[data['temperature'], data['humidity'], data['wind_speed'],
                         wind_encoded, data['wave_height'], data['tide_level'],
                         data['visibility'], data['rainfall'], data['uv_index']]])
    prediction = int(model.predict(features)[0])
    proba = model.predict_proba(features)[0]
    confidence = float(proba[prediction])
    return prediction, confidence

st.set_page_config(page_title="Beach Safety Predictor", page_icon="🏖️", layout="wide")

# Detect theme and set appropriate colors
def is_dark_theme():
    return st.get_option("theme.base") == "dark"

# Dynamic CSS based on theme
def get_theme_styles():
    if is_dark_theme():
        return """
            <style>
            .block-container { padding-top: 1rem !important; }
            .stApp {
                background-image: url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }
            .stApp::before {
                content: "";
                position: fixed;
                top: 0;
                left: 0;
                height: 100%;
                width: 100%;
                background-color: rgba(0, 0, 0, 0.7);
                z-index: -1;
            }
            .weather-box {
                background-color: rgba(40, 40, 40, 0.9);
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.5);
                font-size: 16px;
                color: #ffffff;
            }
            header, footer {visibility: hidden;}
            h1, h2, h3, h4, h5, h6 { color: #ffffff !important; }
            .stNumberInput label, .stButton button { color: #ffffff !important; }
            </style>
        """
    else:
        return """
            <style>
            .block-container { padding-top: 1rem !important; }
            .stApp {
                background-image: url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }
            .stApp::before {
                content: "";
                position: fixed;
                top: 0;
                left: 0;
                height: 100%;
                width: 100%;
                background-color: rgba(255, 255, 255, 0.6);
                z-index: -1;
            }
            .weather-box {
                background-color: rgba(255, 255, 255, 0.9);
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.2);
                font-size: 16px;
            }
            header, footer {visibility: hidden;}
            </style>
        """

st.markdown(get_theme_styles(), unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center;'>
        <h1>🏖️ Beach Safety Predictor</h1>
        <p style='font-size: 18px;'>Check if a beach is safe using live weather data</p>
    </div>
""", unsafe_allow_html=True)

st.subheader("Enter Beach Location")
col1, col2 = st.columns(2)
with col1:
    lat = st.number_input("📍 Latitude", value=19.0760, format="%.6f")
with col2:
    lon = st.number_input("📍 Longitude", value=72.8777, format="%.6f")

if st.button("⚡ Get Beach Safety Prediction"):
    try:
        with st.spinner("Fetching weather data and predicting safety..."):
            data = fetch_weather_data(lat, lon)

            st.markdown(f"""
                <div class="weather-box">
                    <h4>🌤️ Current Weather Details</h4>
                    <ul style="color: {'#ffffff' if is_dark_theme() else '#000000'};">
                        <li>🌡️ <b>Temperature:</b> {data['temperature']:.1f} °C</li>
                        <li>💧 <b>Humidity:</b> {data['humidity']}%</li>
                        <li>💨 <b>Wind Speed:</b> {data['wind_speed']:.1f} km/h ({data['wind_direction']})</li>
                        <li>🌊 <b>Wave Height:</b> {data['wave_height']:.1f} m</li>
                        <li>🌊 <b>Tide Level:</b> {data['tide_level']:.1f} m</li>
                        <li>👁 <b>Visibility:</b> {data['visibility']:.1f} km</li>
                        <li>☔ <b>Rainfall:</b> {data['rainfall']} mm</li>
                        <li>☀ <b>UV Index:</b> {data['uv_index']}</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

            prediction, confidence = predict_beach_safety(data)
            confidence_percent = int(confidence * 100)

            st.subheader("Beach Safety Prediction")

            if prediction == 1:
                st.markdown(f"""
                    <div style="
                        background-color: {'#155724' if is_dark_theme() else '#d4edda'};
                        border-left: 6px solid #28a745;
                        padding: 1rem;
                        border-radius: 10px;
                        margin-bottom: 1rem;
                        font-size: 20px;
                        color: {'#ffffff' if is_dark_theme() else '#000000'};">
                        ✅ <strong>Safe to Visit</strong><br>Enjoy your beach time! 🏖️😎
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style="
                        background-color: {'#721c24' if is_dark_theme() else '#f8d7da'};
                        border-left: 6px solid #dc3545;
                        padding: 1rem;
                        border-radius: 10px;
                        margin-bottom: 1rem;
                        font-size: 20px;
                        color: {'#ffffff' if is_dark_theme() else '#000000'};">
                        ⚠️ <strong>Not Safe to Visit</strong><br>Better to stay away today. 🚫🌊
                    </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
                <div style="
                    background-color: {'#1e2b3c' if is_dark_theme() else '#f0f8ff'};
                    padding: 1rem;
                    border-radius: 12px;
                    border-left: 6px solid {'#2ecc71' if confidence_percent >= 70 else '#f1c40f' if confidence_percent >= 40 else '#e74c3c'};
                    box-shadow: 0 4px 8px rgba(0,0,0,0.05);
                    margin-top: 20px;
                    color: {'#ffffff' if is_dark_theme() else '#000000'};">
                    <h4 style="margin-bottom: 0.5rem;">🎯 Confidence Score</h4>
                    <div style="background-color: {'#2d3748' if is_dark_theme() else '#e0e0e0'}; border-radius: 20px; height: 20px; width: 100%; overflow: hidden;">
                        <div style="
                            height: 100%;
                            width: {confidence_percent}%;
                            background: linear-gradient(to right, #00c853, #b2ff59);
                            border-radius: 20px;">
                        </div>
                    </div>
                    <p style="margin-top: 0.5rem;"><b>{confidence_percent}%</b> confidence in prediction</p>
                </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please check the coordinates and try again.")

show_eval = st.checkbox("Show Model Evaluation Metrics")
if show_eval:
    try:
        df = pd.read_csv("beach_conditions_dataset.csv")
        df['wind_direction'] = df['wind_direction'].map(wind_dir_map)
        y_true = df['is_safe']
        X = df.drop(columns=['date_time', 'is_safe'])
        y_pred = model.predict(X)

        col1, col2, col3 = st.columns([1.2, 1, 1.3])

        with col1:
            st.markdown("### 📊 Classification Report")
            report = classification_report(y_true, y_pred, output_dict=True)
            df_report = pd.DataFrame(report).transpose()
            st.dataframe(df_report.style.format(precision=2).background_gradient(cmap='YlGnBu'), use_container_width=True)

        with col2:
            st.markdown("### 📉 Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Not Safe', 'Safe'], yticklabels=['Not Safe', 'Safe'], ax=ax)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)

        with col3:
            st.markdown("### 🌐 Feature Importance")
            if hasattr(model, "feature_importances_"):
                feature_names = ['temperature', 'humidity', 'wind_speed', 'wind_direction',
                                 'wave_height', 'tide_level', 'visibility', 'rainfall', 'uv_index']
                importances = model.feature_importances_
                fig3 = go.Figure(data=[go.Pie(
                    labels=feature_names,
                    values=importances,
                    hole=0.2,
                    textinfo='label+percent'
                )])
                fig3.update_layout(
                    height=500,
                    margin=dict(t=0, b=30, l=10, r=10),
                    legend=dict(orientation="v", x=1, y=1),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white' if is_dark_theme() else 'black')
                )
                st.plotly_chart(fig3, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not evaluate model: {str(e)}")
