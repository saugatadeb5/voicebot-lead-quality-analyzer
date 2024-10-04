import streamlit as st
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor, as_completed
import string
import re
import time
import matplotlib.pyplot as plt


# Initialize Translator and FinBERT Model
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
    'unfortunately', 'never', 'nai', 'won\'t', 'nahi ji', 'wont'
]

payment_intent_phrases = [
    'pay', 'payment', 'paying', 'i am paying now', 'will pay', 
    'intend to pay', 'promise to pay', 'want to pay', 'am paying','already paid'
]

quality_reason_keywords = [
    'hospitalised', 'financial issue', 'job loss', 'poor family condition',
    'medical issue', 'sick', 'unemployed', 'family crisis', 'financial difficulty',
    'cow ran away','divorce','marriage','fund problem', 'fund issue','will deposit','multiple loans'
]

non_quality_indicators = [
    'will not pay', 'unable to pay', 'cannot afford', 'not interested', 
    'decline', 'reject', 'no payment', 'not going to pay', 'cant pay', 'wont pay', 'not paying','हैलो'
]

date_synonyms = [
    'yesterday', 'today', 'tomorrow', 'day after tomorrow', 'next week',
    'next month', 'call me on', 'reach me on', 'available on', 'please contact on'
]

# Regex to capture common date patterns (e.g., 25th, 2024-10-03, etc.)
date_patterns = [
    r'\b\d{1,2}(st|nd|rd|th)?\b',  # Matches ordinal dates like 25th, 1st, 2nd
    r'\b\d{4}-\d{2}-\d{2}\b',      # Matches dates in YYYY-MM-DD format
    r'\b\d{2}/\d{2}/\d{4}\b',      # Matches dates in MM/DD/YYYY format
    r'\b\d{1,2}/\d{1,2}\b'         # Matches MM/DD or DD/MM format
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

def contains_dates(text):
    """Check if the text contains date-related keywords or matches date patterns."""
    if contains_keywords(text, date_synonyms):
        return True
    
    for pattern in date_patterns:
        if re.search(pattern, text):
            return True
    
    return False

availability_related_phrases = [
    'not at home', 'outside', 'not available', 'is not here', 
    'he is not available', 'she is not available', 'he is outside', 
    'she is outside', 'my husband is not at home', 'my son is not at home', 
    'my daughter is not at home'
]

negotiation_phrases = [
    'asking too much', 'you are asking', 'I have taken', 'I took', 'negotiation', 
    'bargaining', 'I owe less', 'you are charging', 'reduce the amount', 
    'can you reduce', 'asking for more', 'demanding too much', 'not fair price'
]

def translate_and_classify(text):
    """Translate text to English, preprocess, and classify sentiment into High, Medium, or Low Quality Lead."""
    if not text.strip():
        return 'No Utterance Available'
    
    try:
        translated_text = translator.translate(text, source='auto', target='en')
        if not translated_text:
            return 'Medium Quality Lead'
        
        cleaned_text = preprocess_text(translated_text)

        if contains_dates(cleaned_text):
            return 'High Quality Lead'

        if contains_keywords(cleaned_text, availability_related_phrases):
            return 'Medium Quality Lead' 

        if contains_keywords(cleaned_text, negotiation_phrases):
            return 'Medium Quality Lead' 

        if contains_keywords(cleaned_text, non_quality_indicators):
            if contains_keywords(cleaned_text, quality_reason_keywords):
                return 'Medium Quality Lead'
            return 'Low Quality Lead'

        if contains_keywords(cleaned_text, payment_intent_phrases):
            return 'High Quality Lead'

        if contains_keywords(cleaned_text, positive_synonyms):
            if contains_dates(cleaned_text) or contains_keywords(cleaned_text, quality_reason_keywords):
                return 'High Quality Lead'

        if contains_keywords(cleaned_text, quality_reason_keywords):
            return 'Medium Quality Lead'

        return 'Low Quality Lead'

    except Exception as e:
        print(f"Exception occurred: {e}")
        return 'Low Quality Lead'


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
    <h1 class="stTitle">🚀 AI-Powered Lead Quality Analyzer</h1>
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


if 'processed_df' not in st.session_state:
    st.session_state['processed_df'] = None

import streamlit as st
import pandas as pd
import plotly.express as px

# Check if file is uploaded
if uploaded_file is not None:
    if st.session_state.get('uploaded_filename') != uploaded_file.name:
        st.session_state['processed_df'] = None
        st.session_state['uploaded_filename'] = uploaded_file.name

    df = pd.read_csv(uploaded_file)

    if 'utterance' not in df.columns:
        st.error("The dataset must contain an 'utterance' column.")
    else:
        st.success("File successfully uploaded. Preparing to analyze...")

        # Process dataframe if not already processed
        if st.session_state['processed_df'] is None:
            num_rows = df.shape[0]
            num_columns = df.shape[1]
            st.write(f"Number of Rows: {num_rows}")
            st.write(f"Number of Columns: {num_columns}")
            st.write("Dataset Preview:")
            st.dataframe(df.head())

            # Classify the lead quality based on utterances
            utterances = df['utterance'].tolist()
            df['lead_quality'] = classify_texts_in_batches(utterances)

            st.session_state['processed_df'] = df
        else:
            df = st.session_state['processed_df']

        # Calculate lead distribution: count and percentage
        lead_counts = df['lead_quality'].value_counts()
        lead_percentage = df['lead_quality'].value_counts(normalize=True) * 100

        # Create a new dataframe to show counts and percentages
        lead_distribution_df = pd.DataFrame({
            'Lead Quality': lead_counts.index,
            'Count': lead_counts.values,
            'Percentage': lead_percentage.round(2).values  # Two decimal places
        })

        st.write("Lead Quality Distribution:")
        st.dataframe(lead_distribution_df)

        # Plotly bar chart for better visualization
        bar_chart = px.bar(
            lead_distribution_df, 
            x='Lead Quality', 
            y='Count', 
            title='Lead Quality Distribution (Count)', 
            text='Count', 
            animation_frame=None,  # Add animation_frame if needed
            height=300,  # Adjust height for compactness
            width=500   # Adjust width for compactness
        )
        bar_chart.update_traces(textposition='outside', marker_color=['#4CAF50', '#FF9800', '#F44336'])
        bar_chart.update_layout(font=dict(size=10), margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(bar_chart)

        # Plotly pie chart for better percentage visualization
        pie_chart = px.pie(
            lead_distribution_df, 
            values='Percentage', 
            names='Lead Quality', 
            title='Lead Quality Distribution (Percentage)', 
            hole=0.3,  # Creates a donut chart for a cleaner look
            height=300,  # Adjust height for compactness
            width=400   # Adjust width for compactness
        )
        pie_chart.update_traces(textinfo='percent+label', textfont_size=10, marker=dict(colors=['#4CAF50', '#FF9800', '#F44336']))
        pie_chart.update_layout(font=dict(size=10), margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(pie_chart)

        # Download classified leads
        output_file = 'classified_leads.csv'
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Classified Leads", data=csv_data, file_name=output_file, mime='text/csv')

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