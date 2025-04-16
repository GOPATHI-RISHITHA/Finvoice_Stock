import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import speech_recognition as sr
from sklearn.preprocessing import MinMaxScaler

# Localization dictionaries for different languages
translations = {
    "en": {
        "title": "📱 Stock Predictor App",
        "select_language": "Select Language",
        "listening": "🎙 Listening in en... please speak now.",
        "voice_captured": "✅ Voice captured!",
        "company_said": "🗣 You said: `{}`",
        "data_for": "📄 Data for: {}",
        "last_close": "📉 Last Close: ₹{:.2f}",
        "predicted_next": "📈 Predicted Next: ₹{:.2f}",
        "investment_suggestion": "💡 Investment Suggestion",
        "strong_buy": "🔼 Strong Buy (↑ {:.2f}%)",
        "cautious_buy": "🟡 Cautious Buy (↑ {:.2f}%)",
        "hold": "⚖️ Hold ({:.2f}%)",
        "fall_expected": "🔻 Fall Expected (↓ {:.2f}%)",
        "suggested_investment": "💰 Suggested Investment: ₹{}",
        "estimated_profit": "📈 Estimated Profit: ₹{:.2f}",
        "risk_level": "🟢 Risk Level: Low",
        "cautious_risk_level": "🟡 Risk Level: Moderate",
        "medium_risk_level": "🟠 Risk Level: Medium",
        "high_risk_level": "🔴 Risk Level: High",
        "error_company_not_found": "❌ Company not found. Try using keywords like HDFC, TCS, etc.",
        "click_to_speak": "🎤 Click to Speak"
    },
    "te": {
        "title": "📱 స్టాక్ ప్రిడిక్టర్ యాప్",
        "select_language": "భాష ఎంచుకోండి",
        "listening": "🎙 te లో వినిపిస్తోంది... దయచేసి ఇప్పుడు మాట్లాడండి.",
        "voice_captured": "✅ ఆడియో క్యాప్చర్ అయింది!",
        "company_said": "🗣 మీరు చెప్పినది: `{}`",
        "data_for": "📄 డేటా: {}",
        "last_close": "📉 చివరి క్లోజ్: ₹{:.2f}",
        "predicted_next": "📈 అంచనా ప్రైవేట్: ₹{:.2f}",
        "investment_suggestion": "💡 పెట్టుబడి సలహా",
        "strong_buy": "🔼 బలమైన కొనుగోలు (↑ {:.2f}%)",
        "cautious_buy": "🟡 జాగ్రత్త కొనుగోలు (↑ {:.2f}%)",
        "hold": "⚖️ పట్టుకోండి ({:.2f}%)",
        "fall_expected": "🔻 పడిపోతుందని అంచనా (↓ {:.2f}%)",
        "suggested_investment": "💰 సూచించిన పెట్టుబడి: ₹{}",
        "estimated_profit": "📈 అంచనా లాభం: ₹{:.2f}",
        "risk_level": "🟢 ప్రమాదం స్థాయి: తక్కువ",
        "cautious_risk_level": "🟡 ప్రమాదం స్థాయి: మధ్యస్థ",
        "medium_risk_level": "🟠 ప్రమాదం స్థాయి: మధ్య",
        "high_risk_level": "🔴 ప్రమాదం స్థాయి: అధిక",
        "error_company_not_found": "❌ కంపెనీ కనుగొనబడలేదు. HDFC, TCS వంటి కీవర్డ్స్ ఉపయోగించండి.",
        "click_to_speak": "🎤 మాట్లాడడానికి క్లిక్ చేయండి"
    },
    "hi": {
        "title": "📱 स्टॉक प्रेडिक्टर ऐप",
        "select_language": "भाषा चुनें",
        "listening": "🎙 hi में सुन रहा हूँ... कृपया अब बोलें।",
        "voice_captured": "✅ आवाज़ कैप्चर हो गई!",
        "company_said": "🗣 आपने कहा: `{}`",
        "data_for": "📄 डेटा के लिए: {}",
        "last_close": "📉 आखिरी बंद: ₹{:.2f}",
        "predicted_next": "📈 अगला अनुमानित: ₹{:.2f}",
        "investment_suggestion": "💡 निवेश सुझाव",
        "strong_buy": "🔼 मजबूत खरीदें (↑ {:.2f}%)",
        "cautious_buy": "🟡 सतर्क खरीदें (↑ {:.2f}%)",
        "hold": "⚖️ होल्ड करें ({:.2f}%)",
        "fall_expected": "🔻 गिरावट की उम्मीद (↓ {:.2f}%)",
        "suggested_investment": "💰 सुझाई गई निवेश राशि: ₹{}",
        "estimated_profit": "📈 अनुमानित लाभ: ₹{:.2f}",
        "risk_level": "🟢 जोखिम स्तर: कम",
        "cautious_risk_level": "🟡 जोखिम स्तर: मध्यम",
        "medium_risk_level": "🟠 जोखिम स्तर: उच्च",
        "high_risk_level": "🔴 जोखिम स्तर: बहुत उच्च",
        "error_company_not_found": "❌ कंपनी नहीं मिली। HDFC, TCS जैसे कीवर्ड्स का उपयोग करें।",
        "click_to_speak": "🎤 बोलने के लिए क्लिक करें"
    },
    "ta": {
        "title": "📱 ஸ்டாக் ப்ரிடிக்டர் ஆப்",
        "select_language": "மொழி தேர்ந்தெடுக்கவும்",
        "listening": "🎙 ta இல் கேட்கின்றேன்... தயவுசெய்து இப்போது பேசவும்.",
        "voice_captured": "✅ குரல் கைப்பற்றப்பட்டது!",
        "company_said": "🗣 நீங்கள் சொன்னது: `{}`",
        "data_for": "📄 தரவு: {}",
        "last_close": "📉 கடைசியில் மூடிய விலை: ₹{:.2f}",
        "predicted_next": "📈 அடுத்த முன்னறிவு: ₹{:.2f}",
        "investment_suggestion": "💡 முதலீட்டு பரிந்துரை",
        "strong_buy": "🔼 வலுவான வாங்க (↑ {:.2f}%)",
        "cautious_buy": "🟡 எச்சரிக்கையுடன் வாங்க (↑ {:.2f}%)",
        "hold": "⚖️ பிடித்து வைக்கவும் ({:.2f}%)",
        "fall_expected": "🔻 வீழ்ச்சி எதிர்பார்க்கப்படுகிறது (↓ {:.2f}%)",
        "suggested_investment": "💰 பரிந்துரைக்கப்பட்ட முதலீடு: ₹{}",
        "estimated_profit": "📈 மதிப்பிடப்பட்ட லாபம்: ₹{:.2f}",
        "risk_level": "🟢 ஆபத்து நிலை: குறைந்தது",
        "cautious_risk_level": "🟡 ஆபத்து நிலை: மிதமானது",
        "medium_risk_level": "🟠 ஆபத்து நிலை: மத்திய",
        "high_risk_level": "🔴 ஆபத்து நிலை: அதிக",
        "error_company_not_found": "❌ நிறுவனம் கண்டுபிடிக்கவில்லை. HDFC, TCS போன்ற முக்கிய வார்த்தைகளை பயன்படுத்தவும்.",
        "click_to_speak": "🎤 பேச கிளிக் செய்யவும்"
    },
    "kn": {
        "title": "📱 ಸ್ಟಾಕ್ ಪ್ರಿಡಿಕ್ಟರ್ ಆಪ್",
        "select_language": "ಭಾಷೆ ಆಯ್ಕೆಮಾಡಿ",
        "listening": "🎙 kn ನಲ್ಲಿ ಕೇಳುತ್ತಿದ್ದೇನೆ... ದಯವಿಟ್ಟು ಈಗ ಮಾತನಾಡಿ.",
        "voice_captured": "✅ ಧ್ವನಿ ಕ್ಯಾಪ್ಚರ್ ಆಗಿದೆ!",
        "company_said": "🗣 ನೀವು ಹೇಳಿದವು: `{}`",
        "data_for": "📄 ಡೇಟಾ: {}",
        "last_close": "📉 ಕೊನೆಯ ಕ್ಲೋಸ್: ₹{:.2f}",
        "predicted_next": "📈 ಮುಂದಿನ ಅನುವಾದಿತ: ₹{:.2f}",
        "investment_suggestion": "💡 ಹೂಡಿಕೆ ಸಲಹೆ",
        "strong_buy": "🔼 ಬಲವಾದ ಖರೀದಿ (↑ {:.2f}%)",
        "cautious_buy": "🟡 ಎಚ್ಚರಿಕೆಯಿಂದ ಖರೀದಿ (↑ {:.2f}%)",
        "hold": "⚖️ ಹಿಡಿದುಕೊಳ್ಳಿ ({:.2f}%)",
        "fall_expected": "🔻 ಕೆಳಗೆ ಇಳಿಕೆಯಾಗುವುದಾಗಿ ಊಹಿಸಲಾಗಿದೆ (↓ {:.2f}%)",
        "suggested_investment": "💰 ಸೂಚಿಸಲಾಗಿರುವ ಹೂಡಿಕೆ: ₹{}",
        "estimated_profit": "📈 ಅಂದಾಜು ಲಾಭ: ₹{:.2f}",
        "risk_level": "🟢 ಅಪಾಯ ಮಟ್ಟ: ಕಡಿಮೆ",
        "cautious_risk_level": "🟡 ಅಪಾಯ ಮಟ್ಟ: ಮಧ್ಯಮ",
        "medium_risk_level": "🟠 ಅಪಾಯ ಮಟ್ಟ: ಮಧ್ಯ",
        "high_risk_level": "🔴 ಅಪಾಯ ಮಟ್ಟ: ಹೆಚ್ಚು",
        "error_company_not_found": "❌ ಕಂಪನಿಯನ್ನು ಕಂಡುಹಿಡಿಯಲಿಲ್ಲ. HDFC, TCS ಎಂಬ ಕೀವರ್ಡ್‌ಗಳನ್ನು ಬಳಸಿ.",
        "click_to_speak": "🎤 ಮಾತನಾಡಲು ಕ್ಲಿಕ್ ಮಾಡಿ"
    },
    "mr": {
        "title": "📱 स्टॉक प्रेडिक्टर अ‍ॅप",
        "select_language": "भाषा निवडा",
        "listening": "🎙 mr मध्ये ऐकत आहे... कृपया आता बोला.",
        "voice_captured": "✅ आवाज कॅप्चर झाला!",
        "company_said": "🗣 तुम्ही म्हणालात: `{}`",
        "data_for": "📄 डेटा: {}",
        "last_close": "📉 शेवटची बंद: ₹{:.2f}",
        "predicted_next": "📈 पुढील अन्दाज: ₹{:.2f}",
        "investment_suggestion": "💡 गुंतवणूक सल्ला",
        "strong_buy": "🔼 मजबूत खरेदी (↑ {:.2f}%)",
        "cautious_buy": "🟡 काळजीपूर्वक खरेदी (↑ {:.2f}%)",
        "hold": "⚖️ होल्ड करा ({:.2f}%)",
        "fall_expected": "🔻 घसरण अपेक्षित आहे (↓ {:.2f}%)",
        "suggested_investment": "💰 सुचवलेली गुंतवणूक: ₹{}",
        "estimated_profit": "📈 अंदाजित नफा: ₹{:.2f}",
        "risk_level": "🟢 जोखीम स्तर: कमी",
        "cautious_risk_level": "🟡 जोखीम स्तर: मध्यम",
        "medium_risk_level": "🟠 जोखीम स्तर: उच्च",
        "high_risk_level": "🔴 जोखीम स्तर: अत्यधिक",
        "error_company_not_found": "❌ कंपनी सापडली नाही. HDFC, TCS सारख्या किव्हर्ड्स वापरा.",
        "click_to_speak": "🎤 बोलण्यासाठी क्लिक करा"
    }
}



# Function to simulate stock prediction and generate a graph
def simulate_stock_prediction():
    # Simulate some stock data for the sake of demo
    dates = pd.date_range(start="2023-01-01", periods=100)
    close_prices = np.random.randn(100).cumsum() + 1500
    data = pd.DataFrame({'Date': dates, 'Close': close_prices})

    # Plot the stock prices
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data['Close'], color='blue', label='Historical Prices')
    plt.title('Stock Price Prediction Simulation')
    plt.xlabel('Date')

    plt.ylabel('Price')
    plt.legend()

    st.pyplot(plt)  # Show the plot in Streamlit

    # Return the last close price and a simulated predicted price
    return data['Close'].iloc[-1], data['Close'].iloc[-1] + np.random.uniform(10, 20)

# Function to recognize speech input
def recognize_speech(language_code):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write(translations[language_code]['listening'])
        audio = recognizer.listen(source)
        st.write(translations[language_code]['voice_captured'])
    try:
        query = recognizer.recognize_google(audio, language=language_code)
        st.write(translations[language_code]['company_said'].format(query))
        return query
    except sr.UnknownValueError:
        st.write(translations[language_code]['error_company_not_found'])
        return None


# Function to match company files based on the recognized speech
def match_company_files(query, language_code):
    mapping = {
        "en": {
            "hdfc": "HDFC_Bank.csv",
            "tcs": "TCS.csv",
            "reliance": "RELIANCE.csv",
            "infosys": "INFOSYS.csv",
            "hindustan": "HINDUSTAN.csv",
            "itc": "ITC.csv"
        },
        "te": {  # Telugu transliterations
            "టిసిఎస్": "TCS.csv",
            "హెచ్డిఎఫ్సి": "HDFC_Bank.csv",
            "రిలయన్స్": "RELIANCE.csv",
            "ఇన్ఫోసిస్": "INFOSYS.csv",
            "ఇటిసి": "ITC.csv"
        },
        "hi": {  # Hindi transliterations
            "टीसीएस": "TCS.csv",
            "एचडीएफसी": "HDFC_Bank.csv",
            "रिलायंस": "RELIANCE.csv",
            "इंफोसिस": "INFOSYS.csv",
            "आईटीसी": "ITC.csv"
        },
        "ta": {  # Tamil transliterations
            "tcs": "TCS.csv",
            "ஹெச்டிஎஃப்சி": "HDFC_Bank.csv",
            "ரிலையன்ஸ்": "RELIANCE.csv",
            "இன்ஃபோசிஸ்": "INFOSYS.csv",
            "ஐடி": "ITC.csv"
        },
        "kn": {  # Kannada transliterations
            "tcs": "TCS.csv",
            "ಹೆಚ್‌ಡಿಎಫ್‌ಸಿ": "HDFC_Bank.csv",
            "ರಿಲಯನ್ಸ್": "RELIANCE.csv",
            "ಇನ್ಫೋಸಿಸ್": "INFOSYS.csv",
            "ಐಟಿಸಿ": "ITC.csv"
        },
        "mr": {  # Marathi transliterations
            "tcs": "TCS.csv",
            "एचडीएफसी": "HDFC_Bank.csv",
            "रिलायन्स": "RELIANCE.csv",
            "इंफोसिस": "INFOSYS.csv",
            "आयटीसी": "ITC.csv"
        }
    }

    matched = []
    query = query.strip().lower()  # Ensure it's lowercase for easier matching

    # Check for matches in the language-specific dictionary
    for name, filename in mapping.get(language_code, {}).items():
        if name in query:
            matched.append((name.upper(), filename))
    return matched


# Set up Streamlit UI for language selection
st.markdown(f"<h1 style='text-align: center;'>{translations['en']['title']}</h1>", unsafe_allow_html=True)
language = st.selectbox(translations['en']['select_language'], ['en', 'te', 'hi','ta','kn','mr'])

# Add button for voice input
if st.button(translations[language]['click_to_speak']):
    # Capture company name via speech recognition
    company_name = recognize_speech(language)

    if company_name:
        # Match the company name to the corresponding stock file
        matched_files = match_company_files(company_name, language)

        if matched_files:
            for matched in matched_files:
                company_name, filename = matched
                st.write(translations[language]['data_for'].format(company_name))

                # Simulate stock prediction for the recognized company
                last_close, predicted_price = simulate_stock_prediction()

                # Display prediction details in the selected language
                st.write(translations[language]['last_close'].format(last_close))
                st.write(translations[language]['predicted_next'].format(predicted_price))

                # Simulate investment suggestion
                investment_suggestion = translations[language]['cautious_buy'].format(np.random.uniform(0.5, 1.5))
                suggested_investment = 5000
                estimated_profit = np.random.uniform(40, 100)
                risk_level = translations[language]['cautious_risk_level']

                # Show investment suggestion and predicted profit
                st.write(translations[language]['investment_suggestion'])
                st.write(investment_suggestion)
                st.write(translations[language]['suggested_investment'].format(suggested_investment))
                st.write(translations[language]['estimated_profit'].format(estimated_profit))
                st.write(translations[language]['risk_level'])
                st.write(risk_level)
        else:
            st.write(translations[language]['error_company_not_found'])