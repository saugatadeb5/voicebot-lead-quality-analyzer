import streamlit as st
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor, as_completed
import string
import re
import time

translator = GoogleTranslator()
model_name = 'yiyanghkust/finbert-tone'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_pipeline = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, truncation=True, padding=True, max_length=512)

# Define synonyms and keywords
positive_synonyms = [
    'yes', 'yea', 'yep', 'yeah', 'yup', 'sure', 'certainly', 'absolutely',
    'definitely', 'of course', 'okay', 'appreciate', 'pleased', 'happy',
    'satisfied', 'will pay', 'nice', 'haan ji'
]

negative_synonyms = [
    'no', 'nay', 'nope', 'not', 'negative', 'against', 'disagree', 
    'unfortunately', 'never', 'nai', 'won\'t', 'nahi ji','wont'
]

payment_intent_phrases = [
    'pay', 'payment', 'paying', 'i am paying now', 'will pay', 
    'intend to pay', 'promise to pay', 'want to pay','paying'
]

quality_reason_keywords = [
    'hospitalised', 'financial issue', 'job loss', 'poor family condition',
    'medical issue', 'sick', 'unemployed', 'family crisis', 'financial difficulty','cow left way'
]

non_quality_indicators = [
    'will not pay', 'unable to pay', 'cannot afford', 'not interested', 
    'decline', 'reject', 'no payment', 'not going to pay', 'cant pay','wont pay','not paying'
]

# Preprocessing and Classification Functions
def preprocess_text(text):
    """Preprocess the text: remove punctuation, convert to lowercase, strip whitespace."""
    remove_chars = string.punctuation + '.,;'
    trans_table = str.maketrans('', '', remove_chars)
    return text.translate(trans_table).strip().lower()

def contains_keywords(text, keywords):
    """Check if the text contains any keywords from the list."""
    return any(keyword in text for keyword in keywords)

def translate_and_classify(text):
    """Translate text to English, preprocess, and classify sentiment."""
    if not text.strip():
        return 'No Utterance Available'
    
    try:
        translated_text = translator.translate(text, source='auto', target='en')
        if not translated_text:
            return 'Non-Quality Lead'

        cleaned_text = preprocess_text(translated_text)

        # Check for non-quality indicators
        if contains_keywords(cleaned_text, non_quality_indicators):
            return 'Non-Quality Lead'
        
        # Check for quality reason indicators or positive payment intent
        if contains_keywords(cleaned_text, quality_reason_keywords) or \
           contains_keywords(cleaned_text, payment_intent_phrases):
            return 'Quality Lead'

        # Additional check for direct payment refusal phrases
        if "won\'t pay" in cleaned_text or 'cannot pay' in cleaned_text:
            return 'Non-Quality Lead'

        result = sentiment_pipeline(cleaned_text)
        label = result[0]['label']
        return 'Quality Lead' if label == 'POSITIVE' else 'Non-Quality Lead'
    
    except Exception as e:
        print(f"Exception occurred: {e}")
        return 'Non-Quality Lead'



def classify_texts_in_batches(texts, batch_size=100, num_workers=4):
    """Classify a batch of texts using a pre-trained model."""
    all_sentiments = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(translate_and_classify, text): text for text in texts}
        progress_bar = st.progress(0)
        total_futures = len(futures)

        for i, future in enumerate(as_completed(futures)):
            try:
                sentiment = future.result()
                all_sentiments.append(sentiment)
            except Exception as e:
                print(f"Error during classification: {e}")
                all_sentiments.append('No Utterance Available')
            progress_bar.progress((i + 1) / total_futures)
    
    return all_sentiments

# Streamlit UI
st.set_page_config(page_title="AI-Powered Lead Quality Analyzer", layout="wide")

st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background: linear-gradient(135deg, #f5f5f5, #e3e3e3);
            color: #333;
        }
        .css-1a4u04r { /* Header */
            color: #8033ff;
        }
        .stTitle {
            font-size: 3em;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
            margin-bottom: 20px;
            animation: fadeIn 2s;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            text-align: center;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 5px;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #45a049;
            transform: scale(1.05);
            transition: transform 0.2s;
        }
        .stTextArea>div>textarea {
            border-color: #4CAF50;
            border-radius: 5px;
            background-color: #ffffff;
            transition: box-shadow 0.3s;
        }
        .stTextArea>div>textarea:focus {
            box-shadow: 0 0 5px rgba(76, 175, 80, 0.5);
            outline: none;
        }
        .stProgress>div {
            background-color: #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <h1 class="stTitle">🚀 LLM-Powered Lead Quality Analyzer</h1>
    <p style="text-align: center; font-size: 1.2em; color: #555;">
        Welcome to the AI-Powered Lead Quality Analyzer! 🌟 This tool utilizes FinBERT for customer sentiment analysis.
    </p>
    <div style="text-align: center;">
        <img src="https://media.licdn.com/dms/image/v2/D560BAQGztDL9Ut7okA/company-logo_200_200/company-logo_200_200/0/1708509740424/dpdzero_logo?e=2147483647&v=beta&t=A-fO6oYLlIpYF5P57qnf490Ekxnn0z8HMBo2gJu9jfo" alt="Company Logo" style="width: 200px;"/>
    </div>
""", unsafe_allow_html=True)

st.header("Upload Your CSV File")
st.markdown("""
    <p style="font-size: 1.1em; color: #555;">
        Please upload a CSV file containing the 'utterance' column. The file will be processed to analyze lead quality based on sentiment analysis.
    </p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="collapsed")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.write("Columns found in the dataset:", df.columns)
    
    if 'utterance' not in df.columns:
        st.error("The dataset must contain an 'utterance' column.")
    else:
        st.success("File successfully uploaded. Preparing to analyze...")
        st.write("Click below to start the classification process.")
        
        if st.button("Start Classification"):
            with st.spinner('Processing your file...'):
                utterances = df['utterance'].tolist()
                df['sentiment'] = classify_texts_in_batches(utterances)

                df['lead_quality'] = df['sentiment'].apply(lambda x: 'Quality Lead' if x == 'Quality Lead' else 'Non-Quality Lead')
                df.drop(columns=['sentiment'], inplace=True)
                
                output_file = 'classified_leads.csv'
                df.to_csv(output_file, index=False)
                
                st.success("Classification complete!")
                st.write("Download your results below:")
                st.download_button(
                    label="Download Classified Leads",
                    data=open(output_file, 'rb').read(),
                    file_name=output_file,
                    mime='text/csv',
                    key='download_button'
                )

st.header("Classify Individual Text")
st.markdown("""
    <p style="font-size: 1.1em; color: #555;">
        Enter your text below to classify it individually. The system will analyze the text and provide a classification based on sentiment analysis.
    </p>
""", unsafe_allow_html=True)

input_text = st.text_area("Enter your text here:", placeholder="Type your text here...", height=100)

if st.button("Analyze Text"):
    if input_text:
        message_placeholder = st.empty()
        for i in range(5): 
            message_placeholder.text(f"Analyzing text{'.' * (i % 4)}")
            time.sleep(0.5)

        result = translate_and_classify(input_text)
        message_placeholder.empty() 
        st.write(f"**Classification Result:** {result}")
    else:
        st.warning("Please enter some text to classify.")
