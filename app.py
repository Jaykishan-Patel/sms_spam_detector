
import nltk
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

for pkg in ["punkt", "stopwords", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except:
        nltk.download(pkg)

ps = PorterStemmer()
# Text Transform Function
def transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Load Model + Vectorizer
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Page Title
st.markdown("""
    <h1 style="text-align:center; color:#4CAF50;">SMS / Email Spam Detector</h1>
    <p style="text-align:center; font-size:18px; color:#555;">
        Identify whether a message is spam or not using Machine Learning.
    </p>
""", unsafe_allow_html=True)

# Input Box
input_sms = st.text_area("Enter Message Here:", height=150)

# Predict Button
if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("Please enter a message first.")
    else:
        trans_text = transform(input_sms)
        vector_input = tfidf.transform([trans_text])
        result = model.predict(vector_input)[0]

        if result == 0:
            st.success("✅ The message is **NOT SPAM**")
            st.balloons()
        else:
            st.error("🚨 The message is **SPAM**")

