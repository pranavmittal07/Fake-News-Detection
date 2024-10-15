import streamlit as st
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Ensure stopwords are downloaded
nltk.download('stopwords')

# Load vectorizer and model from disk
@st.cache_resource
def load_resources():
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return vectorizer, model

vectorizer, model = load_resources()

# Define stemming function
ps = PorterStemmer()
def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    content = content.split()
    content = [ps.stem(word) for word in content if word not in stopwords.words('english')]
    return ' '.join(content)

# Website
st.title('Fake News Detector')
input_text = st.text_area('Enter news article')

def predict_fake_news(text):
    processed_text = stemming(text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)
    return prediction[0]

if st.button('Check'):
    if input_text:
        try:
            pred = predict_fake_news(input_text)
            if pred == 1:
                st.write('The news is Fake')
            else:
                st.write('The news is Real')
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning('Please enter some text to check.')

