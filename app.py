# app.py

import streamlit as st
import nltk
import pickle
import re
import string

from nltk.corpus import stopwords

# Ensure nltk dependencies are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load stopwords once
stop_words = set(stopwords.words('english'))

# Load model
try:
    with open('spam_voting_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Could not find the model file 'spam_voting_model.pkl'. Please make sure it's in the app directory.")

# Load vectorizer
try:
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Could not find the vectorizer file 'tfidf_vectorizer.pkl'. Please make sure it's in the app directory.")

# Preprocess function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Streamlit App UI
st.title("📧 Spam Email Classifier")
st.write("Enter a message and check if it's spam or not.")

message = st.text_area("✉️ Enter your email text below:")

if st.button("🔍 Predict"):
    if message.strip() == "":
        st.warning("⚠️ Please enter a message to classify.")
    else:
        cleaned_msg = preprocess(message)
        msg_vec = vectorizer.transform([cleaned_msg])
        prediction = model.predict(msg_vec)

        if prediction[0] == 1:
            st.error("🚫 This is **SPAM**!")
        else:
            st.success("✅ This is **NOT SPAM**.")
