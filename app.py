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

positive_synonyms = [
    'yes', 'yea', 'yep', 'yeah', 'yup', 'sure', 'affirmative', 'certainly', 'absolutely',
    'definitely', 'of course', 'right', 'correct', 'agree', 'okay', 'ok', 'yass', 'yasss',
    'indeed', 'grateful', 'thank you', 'thanks', 'appreciate', 'pleased', 'happy', 'satisfied','will pay','nice',
    'haan ji'  # Romanized Hindi for 'yes'
]

negative_synonyms = [
    'no', 'nay', 'nope', 'not', 'negative', 'against', 'disagree', 'unfortunately','never','nai','never','cant',"wouldn't",'will not',"shouldn't","cannot","would not"
    'nahi ji','wont',"won't"  # Romanized Hindi for 'no'
]

date_keywords = [
    'today', 'tomorrow', 'yesterday', 'date', 'th', 'rd', 'nd', 'day', 'week', 'month', 'year',
    'on', 'at', 'by', 'before', 'after','tommorow','last'
]

simple_phrases = {
    'thanks', 'thank you', 'thank you very much', 'appreciate it', 'thanks a lot', 'thanks so much',
    'many thanks', 'grateful', 'cheers', 'thanks for that', 'thanks a bunch', 'thanks for your help',
    'thanks for your time', 'thank you for your assistance'
}

greetings = {'hello', 'hi', 'hey', 'goodbye', 'morning', 'evening', 'night'}

quality_reason_keywords = [
    'death', 'hospitalised', 'financial issue', 'job loss', 'left job', 'poor family condition','bittiye','medical issue'
    'medical', 'sick', 'unemployed', 'family crisis', 'financial difficulty','give me some time','fund issue','i want some time',
]

payment_methods = [
    'upi', 'online payment', 'credit card', 'debit card', 'bank transfer', 'cash'
]

non_quality_indicators = [
    'will not pay', 'unable to pay', 'cannot afford', 'not interested', 'decline', 'reject',
    'cannot make payment','cant pay','never pay', 'no payment', 'not going to pay','cannot pay'
]

def preprocess_text(text):
    """Preprocess the text: remove punctuation, convert to lowercase, strip whitespace."""
    remove_chars = string.punctuation + '.,;'
    
    trans_table = str.maketrans('', '', remove_chars)
    
    text = text.translate(trans_table).strip().lower()
    return text

def contains_date_or_time(text):
    """Check if the text contains references to dates or times."""
    if any(keyword in text.lower() for keyword in date_keywords):
        return True
    date_patterns = [
        r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
        r'\b\d{2}/\d{2}/\d{4}\b',  # MM/DD/YYYY
        r'\b\d{2}-\d{2}-\d{4}\b',  # DD-MM-YYYY
        r'\b\d{1,2} \w{3,9} \d{4}\b',  # DD Month YYYY
        r'\b\w{3,9} \d{1,2}, \d{4}\b'  # Month DD, YYYY
    ]
    for pattern in date_patterns:
        if re.search(pattern, text):
            return True
    return False

def is_greeting(text):
    """Check if the text is a common greeting."""
    return text in greetings

def contains_positive_synonym(text):
    """Check if the text contains any positive synonym or variant."""
    return any(synonym in text for synonym in positive_synonyms)

def contains_negative_synonym(text):
    """Check if the text contains any negative synonym or variant."""
    return any(synonym in text for synonym in negative_synonyms)

def contains_negative_synonym(text):
    """Check if the text contains any negative synonym or variant."""
    return any(synonym in text for synonym in negative_synonyms)

def is_simple_phrase(text):
    """Check if the text is a simple phrase."""
    return text in simple_phrases

def contains_quality_reason(text):
    """Check if the text contains any keywords related to quality reasons."""
    return any(keyword in text for keyword in quality_reason_keywords)

def contains_non_quality_indicator(text):
    """Check if the text contains phrases indicating non-quality leads."""
    return any(phrase in text for phrase in non_quality_indicators)


def contains_payment_intent(text):
    """Check if the text indicates a payment intent or method."""
    return 'pay' in text or 'payment' in text or any(method in text for method in payment_methods)

def contains_future_payment_commitment(text):
    """Check if the text indicates a future payment commitment."""
    future_phrases = ['will pay', 'will make payment', 'plan to pay', 'intend to pay', 'promise to pay','want to pay']
    return any(phrase in text for phrase in future_phrases)

def translate_and_classify(text):
    """Translate text to English, preprocess, and classify sentiment."""
    if not text.strip():
        return 'No Utterance Available'
    
    try:
        translated_text = translator.translate(text, source='auto', target='en')
        if not translated_text:
            return 'Non-Quality Lead'

        cleaned_text = preprocess_text(translated_text)
        
        if is_greeting(cleaned_text):
            return 'Non-Quality Lead'  
        
        if contains_non_quality_indicator(cleaned_text) and not contains_future_payment_commitment(cleaned_text):
            return 'Non-Quality Lead'

        if contains_negative_synonym(cleaned_text) and contains_non_quality_indicator(cleaned_text):
            return 'Non-Quality Lead'
        
        if contains_negative_synonym(cleaned_text) and contains_payment_intent(cleaned_text):
            return 'Non-Quality Lead'
        
        if contains_quality_reason(cleaned_text):
            return 'Quality Lead'

        if contains_quality_reason(cleaned_text) or contains_positive_synonym(cleaned_text) or contains_date_or_time(cleaned_text):
            return 'Quality Lead'
        
        if contains_quality_reason(cleaned_text) and (contains_negative_synonym(cleaned_text) or contains_date_or_time(cleaned_text) or contains_non_quality_indicator(cleaned_text)):
            return 'Quality Lead'
        
        if is_simple_phrase(cleaned_text):
            return 'Quality Lead'
        
        if contains_future_payment_commitment(cleaned_text):
            return 'Quality Lead'
        
        if contains_future_payment_commitment(cleaned_text) or contains_payment_intent(cleaned_text):
            return 'Quality Lead'
        
        result = sentiment_pipeline(cleaned_text)
        label = result[0]['label']
        if label == 'POSITIVE':
            return 'Quality Lead'
        elif label == 'NEGATIVE':
            return 'Non-Quality Lead'
        else:
            return 'Quality Lead'
    except Exception as e:
        print(f"Exception occurred: {e}")
        return 'Non-Quality Lead'

def classify_texts_in_batches(texts, batch_size=100, num_workers=4):
    """Classify a batch of texts using a pre-trained model with additional domain-specific rules."""
    all_sentiments = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(translate_and_classify, text) for text in texts]
        progress_bar = st.progress(0)
        total_futures = len(futures)
        for i, future in enumerate(as_completed(futures)):
            try:
                sentiment = future.result()
                all_sentiments.append(sentiment)
            except Exception as e:
                print(f"Error during classification: {e}")
                all_sentiments.append('No Utterance Available')
            # Update progress bar
            progress_bar.progress((i + 1) / total_futures)
    
    return all_sentiments

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
        .css-1c6d7x6 { /* Text Area */
            background-color: #ffffff;
            color: #333;
        }
        .stTitle {
            font-size: 3em;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
            margin-bottom: 20px;
        }
        .stHeader {
            font-size: 2em;
            color: #333;
            font-weight: bold;
        }
        .stTextArea>div>label, .stTextArea>div>textarea {
            color: #333;
            font-weight: bold;
        }
        .stTextArea>div>textarea {
            border-color: #4CAF50;
            border-radius: 5px;
            background-color: #ffffff;
        }
        .stTextInput>div>label, .stFileUploader>div>label, .stButton>button, .stDownloadButton>button {
            color: #4CAF50;
            font-weight: bold;
        }
        .stFileUploader>div {
            border: 2px dashed #4CAF50;
            border-radius: 5px;
            padding: 20px;
            text-align: center;
        }
        .stProgress>div {
            background-color: #4CAF50;
        }
        .stDownloadButton>button {
            background-color: #2196F3;
            color: white;
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 5px;
            transition: background-color 0.3s;
            font-weight: bold;
        }
        .stDownloadButton>button:hover {
            background-color: #0b79d0;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 5px;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 class="stTitle">🚀 LLM-Powered Lead Quality Analyzer</h1>
        <p style="font-size: 1.5em; color: #8033ff;">Welcome to the AI-Powered Lead Quality Analyzer! 🌟</p>
        <p style="font-size: 1.2em; color: #555;">
            This tool leverages FinBERT for sentiment analysis and provides a comprehensive lead quality evaluation.
        </p>
        <img src="https://media.licdn.com/dms/image/v2/D560BAQGztDL9Ut7okA/company-logo_200_200/company-logo_200_200/0/1708509740424/dpdzero_logo?e=2147483647&v=beta&t=A-fO6oYLlIpYF5P57qnf490Ekxnn0z8HMBo2gJu9jfo" alt="Description of Image" style="width: 18%; max-width: 200px;"/>
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
                time.sleep(1)  
                utterances = df['utterance'].tolist()
                df['sentiment'] = classify_texts_in_batches(utterances)

               
                def determine_lead_quality(sentiment):
                    """Determine the lead quality based on sentiment."""
                    if sentiment == 'Quality Lead':
                        return 'Quality Lead'
                    if sentiment == 'Non-Quality Lead':
                        return 'Non-Quality Lead'
                    elif sentiment == 'No Utterance Available':
                        return ''
                    else:
                        return 'Non-Quality Lead'
                
                df['lead_quality'] = df['sentiment'].apply(determine_lead_quality)
                df = df.drop(columns=['sentiment'])
                
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
        st.write("Analizing text...")
        result = translate_and_classify(input_text)
        st.write(f"**Classification Result:** {result}")
    else:
        st.warning("Please enter some text to classify.")
