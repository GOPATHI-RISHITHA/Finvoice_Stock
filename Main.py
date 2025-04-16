# Required libraries
import pandas as pd
import numpy as np
import speech_recognition as sr
from googletrans import Translator
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Input
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Global variables
translator = Translator()
selected_lang_name = ""
target_lang_code = "en"  # Default fallback

# --- Voice Input ---
def get_voice_input():
    global selected_lang_name, target_lang_code
    languages = {
        "1": ("English", "en-IN", "en"),
        "2": ("Hindi", "hi-IN", "hi"),
        "3": ("Telugu", "te-IN", "te"),
        "4": ("Tamil", "ta-IN", "ta"),
        "5": ("Kannada", "kn-IN", "kn"),
        "6": ("Bengali", "bn-IN", "bn"),
    }

    print("\n🌐 Select a language for voice input:")
    for key, (lang, _, _) in languages.items():
        print(f"{key}. {lang}")
    
    choice = input("Enter your choice (e.g., 1 for English): ").strip()

    if choice not in languages:
        print("⚠️ Invalid choice. Defaulting to English.")
        selected_lang_name, _, target_lang_code = languages["1"]
    else:
        selected_lang_name, _, target_lang_code = languages[choice]
        print(f"✅ You selected: {selected_lang_name}")

    r = sr.Recognizer()
    with sr.Microphone() as source:
        print(f"\n🎙 Speak now ({selected_lang_name})...")
        r.pause_threshold = 1.5
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.listen(source, timeout=10, phrase_time_limit=8)

    try:
        query = r.recognize_google(audio, language=languages[choice][1])
        print(f"🗣 You said: {query}")
        
        if target_lang_code != 'en':
            translated = translator.translate(query, src=target_lang_code, dest='en')
            print(f"🌐 Translated to English: {translated.text}")
            return translated.text.lower()
        else:
            return query.lower()
    except Exception as e:
        print(f"⚠️ Error: {str(e)}")
        return ""

# --- Translator print ---
def tprint(text):
    if target_lang_code != 'en':
        try:
            translated = translator.translate(text, src='en', dest=target_lang_code)
            print(translated.text)
        except:
            print(text)
    else:
        print(text)

# --- Company Match ---
def match_company_files(query):
    mapping = {
        "hdfc": "HDFC_Bank.csv",
        "tcs": "TCS.csv",
        "reliance": "RELIANCE.csv",
        "infosys": "INFOSYS.csv",
        "hindustan": "HINDUSTAN.csv",
        "itc": "ITC.csv"
    }
    matched = []
    for name, filename in mapping.items():
        if name in query:
            matched.append((name.upper(), filename))
    return matched

# --- Data Preprocessing ---
def load_and_prepare_data(filename):
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip().str.lower()
    df['date'] = pd.to_datetime(df['date'], format='%d-%b-%Y')
    df['close'] = df['close'].astype(str).str.replace(',', '').astype(float)
    df = df.sort_values('date')

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['close']].values)

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i])
    
    X, y = np.array(X), np.array(y)
    return X, y, scaler, df

# --- Model ---
def build_model():
    model = Sequential()
    model.add(Input(shape=(60, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- Wrapper to reduce retracing ---
@tf.function(reduce_retracing=True)
def make_prediction(model, X_input):
    return model(X_input, training=False)

# --- Investment Suggestion ---
def suggest_investment(last_close, predicted):
    change_percent = ((predicted - last_close) / last_close) * 100
    tprint("\n💡 Investment Suggestion:")
    if change_percent > 2:
        tprint(f"🔼 Strong Buy: Predicted to rise by {change_percent:.2f}%")
        tprint("💰 Suggested Investment: ₹10,000")
        est_profit = (predicted - last_close) * (10000 / last_close)
        tprint(f"📈 Estimated Profit: ₹{est_profit:.2f}")
        tprint("🟢 Risk Level: Low")
    elif 0.5 < change_percent <= 2:
        tprint(f"🟡 Cautious Buy: Small rise of {change_percent:.2f}%")
        tprint("💰 Suggested Investment: ₹5,000")
        est_profit = (predicted - last_close) * (5000 / last_close)
        tprint(f"📈 Estimated Profit: ₹{est_profit:.2f}")
        tprint("🟡 Risk Level: Moderate")
    elif -0.5 <= change_percent <= 0.5:
        tprint(f"⚖️ Hold: Almost no change ({change_percent:.2f}%)")
        tprint("🕒 Suggested Action: Wait and monitor trend.")
        tprint("🟠 Risk Level: Medium")
    else:
        tprint(f"🔻 Fall Expected: Predicted drop of {change_percent:.2f}%")
        tprint("❌ Suggested Action: Avoid investing / Consider selling.")
        tprint("🔴 Risk Level: High")

# --- Main Execution ---
query = get_voice_input()
matched_companies = match_company_files(query)

if matched_companies:
    suggestions = []
    for company_name, company_file in matched_companies:
        tprint(f"\n📄 Loading data for: {company_name}")
        X, y, scaler, df = load_and_prepare_data(company_file)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = build_model()
        model.fit(X, y, epochs=20, batch_size=32, verbose=0)

        last_60 = df['close'].values[-60:]
        last_60_scaled = scaler.transform(last_60.reshape(-1, 1))
        X_test = np.array([last_60_scaled]).reshape(1, 60, 1)

        prediction = make_prediction(model, X_test)
        predicted_price = scaler.inverse_transform(prediction.numpy())[0][0]
        last_close = df['close'].values[-1]
        change_percent = ((predicted_price - last_close) / last_close) * 100

        tprint(f"\n📉 Last close for {company_name}: ₹{last_close:.2f}")
        tprint(f"📈 Predicted next for {company_name}: ₹{predicted_price:.2f}")
        if predicted_price > last_close:
            tprint(f"📊 {company_name} likely to RISE 📈")
        else:
            tprint(f"📊 {company_name} likely to FALL 📉")

        # Graph
        recent_prices = df['close'].values[-50:].tolist()
        predicted_series = [None] * 49 + [last_close, predicted_price]
        plt.figure(figsize=(10, 4))
        plt.plot(recent_prices, label=f"{company_name} Close", marker='o')
        plt.plot(range(49, 51), predicted_series[49:], label="Predicted", marker='x', linestyle='--', color='red')
        plt.title(f"{company_name} - Recent Trend & Prediction")
        plt.xlabel("Days")
        plt.ylabel("Price (₹)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        suggest_investment(last_close, predicted_price)

        suggestions.append({
            "company": company_name,
            "change_percent": change_percent,
            "last_close": last_close,
            "predicted": predicted_price
        })

    # Final Summary
    tprint("\n🧾 Final Recommendation Summary:")
    for s in suggestions:
        tprint(f"- {s['company']}: Change = {s['change_percent']:.2f}% | Last = ₹{s['last_close']:.2f} | Predicted = ₹{s['predicted']:.2f}")

    best = max(suggestions, key=lambda x: x["change_percent"])
    tprint(f"\n✅ Best Option: {best['company']} (↑ {best['change_percent']:.2f}%)")
    if best["change_percent"] < 0:
        tprint("⚠️ However, all options are predicted to fall. Caution advised.")
else:
    tprint("❌ Could not find the company in your question. Try again with keywords like HDFC, ITC, etc.")

