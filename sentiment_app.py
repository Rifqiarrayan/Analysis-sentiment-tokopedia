import streamlit as st
import pandas as pd
import re
import string
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# === FUNGSI PREPROCESSING ===

def cleansing(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # hapus angka dan karakter khusus
    text = re.sub(r'rt|user', '', text)  # hapus kata tertentu
    return text

def remove_stopwords(text):
    stop_factory = StopWordRemoverFactory()
    stopwords = stop_factory.get_stop_words()
    return ' '.join([word for word in text.split() if word not in stopwords])

def stemming(text):
    stem_factory = StemmerFactory()
    stemmer = stem_factory.create_stemmer()
    return stemmer.stem(text)

def preprocess(text):
    text = cleansing(text)
    text = remove_stopwords(text)
    text = stemming(text)
    return text

# === LOAD DATA ===

@st.cache_data
def load_data():
    df = pd.read_excel("Sentiment_analysis_Tokopedia.xlsx")  # Pastikan file ini ada
    return df

# === TRAIN MODEL ===

@st.cache_resource
def train_model(df):
    df['cleaned'] = df['Ulasan_clean'].astype(str).apply(preprocess)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['cleaned'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, vectorizer, accuracy

# === STREAMLIT UI ===

st.title("Analisis Sentimen Tokopedia 🛒")
st.write("Masukkan ulasan produk untuk diklasifikasi sentimennya (positif/negatif).")

df = load_data()
model, vectorizer, acc = train_model(df)

st.write(f"🎯 Akurasi model: **{acc:.2%}**")

user_input = st.text_area("Tulis ulasan produk Tokopedia di sini:")

if st.button("Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("Harap masukkan teks ulasan.")
    else:
        cleaned_input = preprocess(user_input)
        vectorized_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vectorized_input)[0]
        label = "Positif 😊" if prediction == 1 else "Negatif 😞"
        st.success(f"Hasil prediksi: **{label}**")
