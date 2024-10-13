import streamlit as st
import pandas as pd
import re
import string
# import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import joblib

# Load TFIDFVectorizer
tfidf_vectorizer = joblib.load('TFIDF_model.pkl')

# Load your trained model
model_svm = joblib.load('svm_model.pkl')
model_rf = joblib.load('random_forest_model.pkl')

# Function to Preprocessing
# nltk.download('punkt')
# nltk.download('stopwords') 
# nltk.download('wordnet')
# nltk.download('punkt_tab')
common_words = set(['game', 'com', 'unk', 'like'])
stop_words = set(stopwords.words('english')).union(common_words)
lem = WordNetLemmatizer()

def remove_stop_words(txt):
    return ' '.join([x for x in txt.split() if x not in stop_words])
def remove_punc(txt):
    text_non_punct = "".join([char for char in txt if char not in string.punctuation])
    return text_non_punct
def remove_digit(txt):
    text_non_digit = re.sub(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b", '', txt).strip()
    return text_non_digit
def lemmatizing(txt):
    lemmatized = [lem.lemmatize(word, pos='v') for word in txt]
    return lemmatized

def preprocessing(txt):
    txt = txt.lower()
    txt = remove_stop_words(txt)
    txt = remove_punc(txt)
    txt = remove_digit(txt)
    txt_token = word_tokenize(txt)
    txt_lem = lemmatizing(txt_token)
    txt = ' '.join(txt_lem)
    return txt

# Page configuration
st.set_page_config(page_title="Sentiment Analysis App", layout="wide", initial_sidebar_state="collapsed")

# Sidebar information
with st.sidebar:
    st.write("### Model SVM Info:")
    st.write("Sentiment Analysis Model")
    st.write("Train Accuracy: 98%")
    st.write("Test Accuracy: 93%")
    st.write("----------------------")
    st.write("### Model Random Forest Info:")
    st.write("Sentiment Analysis Model")
    st.write("Train Accuracy: 99%")
    st.write("Test Accuracy: 91%")

# Title and description
st.title("Sentiment Analysis")
st.write("Analyze text or upload a file to predict sentiment.")

# Text input for real-time feedback
user_input = st.text_area("Enter text for sentiment analysis:", placeholder="Type here...")

choose_model = st.radio(
    "Choose your model:",
    ('SVM', 'Random Forest'))

if st.button("Analyze Sentiment"):
    if user_input:
        # Preprocess the text using the vectorizer
        user_input = preprocessing(user_input)
        user_input_tfidf = tfidf_vectorizer.transform([user_input])

        # Make the prediction and choose model
        if choose_model == 'SVM':
            prediction = model_svm.predict(user_input_tfidf)
        else:
            prediction = model_rf.predict(user_input_tfidf)

        # Display sentiment with visualization
        if prediction[0] == "Positive":
            st.success("Positive 😊")
        elif prediction[0] == "Negative":
            st.error("Negative 😠")
        elif prediction[0] == "Neutral":
            st.warning("Neutral 😐")
        else:
            st.warning("Irrelevant")


# File upload for batch sentiment analysis
uploaded_file = st.file_uploader("Upload a CSV file for batch sentiment analysis", type=["csv"])
st.write("Data must have a text column with name is 'text' ")

if uploaded_file:
    # Read the file into a DataFrame
    data = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    # Expected column names
    expected_columns = ['text']
    
    if set(expected_columns).issubset(data.columns):
        st.write("The uploaded file has the correct column!")
        # st.write(data.head())
        
        # Make predictions 
        # Transform the text using the TF-IDF vectorizer
        data['text'] = data['text'].apply(lambda x: preprocessing(x))
        text_data_tfidf = tfidf_vectorizer.transform(data['text'])

        # Predict sentiments for the text data
        
        if choose_model == 'SVM':
            predictions = model_svm.predict(text_data_tfidf)
        else:
            predictions = model_rf.predict(text_data_tfidf)

        # Add predictions to the DataFrame
        data['Sentiment'] = predictions 
        
        st.write("### Sentiment Analysis Results:")
        st.write(data[['text', 'Sentiment']])

        # Summary Section
        positive_count = sum(data['Sentiment'] == "Positive")
        neutral_count = sum(data['Sentiment'] == "Neutral")
        negative_count = sum(data['Sentiment'] == "Negative")
        irrelevant_count = sum(data['Sentiment'] == "Irrelevant")
        
        st.write("### Summary:")
        st.write(f"Total texts analyzed: {len(data)}")
        st.write(f"Positive: {positive_count}")
        st.write(f"Neutral: {neutral_count}")
        st.write(f"Negative: {negative_count}")
        st.write(f"Irrelevant: {irrelevant_count}")
    else:
        st.error(f"The uploaded file doesn't have the correct columns. Expected: {expected_columns}, but got: {data.columns.tolist()}")

# User feedback section
st.write("### Feedback")
# feedback = st.text_input("How accurate was the prediction? (1-5)", placeholder='Rating here...')
options = [1, 2, 3, 4, 5]
feedback = st.selectbox("How accurate was the prediction? (1-5)", options, index=None)

if st.button("Submit Feedback"):
    st.write("Thank you for your feedback!")

# Footer or final notes
st.write("App built with Streamlit.")

#streamlit run d:/Data science/Final Project/data/sentiment_app.py