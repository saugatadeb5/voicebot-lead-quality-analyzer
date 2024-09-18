# AI-Powered Lead Quality Analyzer: Architectural Documentation

## Introduction

The **AI-Powered Lead Quality Analyzer** is a designed to classify leads based on textual interactions (utterances) Leveraging the power of **FinBERT: A Large Language Model for Extracting Information from Financial Text**, and incorporating domain-specific heuristics, the application determines whether a lead is of "Quality" or "Non-Quality" based on the content of their communication.

This documentation provides a comprehensive overview of the application's architecture, detailing each component's role and how they interact to deliver accurate lead quality assessments.

### Usage:

### Steps to Download

### **Step 1: Clone the Repository**

1. Open your terminal (Command Prompt, Git Bash, etc.).
2. Clone the repository from GitHub to your local machine.
    
    ```bash
    git clone https://github.com/saugatadeb5/voicebot-lead-quality-analyzer.git
    ```
    

### **Step 2: Navigate to the Project Directory**

1. Change the current directory to the folder where you cloned the repository:
    
    ```bash
    cd voicebot-lead-quality-analyzer
    ```
    

### **Step 3: Create a Virtual Environment (Optional but Recommended)**

1. Create a virtual environment to manage dependencies:
    
    ```bash
    python -m venv venv
    ```
    
2. Activate the virtual environment:
    - On **Windows**:
        
        ```bash
        venv\Scripts\activate
        ```
        
    - On **macOS/Linux**:
        
        ```bash
        source venv/bin/activate
        ```
        

### **Step 4: Install Dependencies**

1. Install the required Python packages using `requirements.txt`:
    
    ```bash
    pip install -r requirements.txt
    ```
    

### **Step 5: Run the Streamlit Application**

1. Start the Streamlit app by running the following command:
    
    ```bash
    streamlit run app.py
    ```
    
2. After running the command, the application will launch in your default web browser, or you can access it by navigating to **http://localhost:8501**.

---

## High-Level Architecture Overview

**Figure**: High-level architecture of the AI-Powered Lead Quality Analyzer.

The application consists of the following key components:

1. **User Interface (UI)**: Built with Streamlit, allowing users to upload CSV files or input individual texts.
2. **Data Preprocessing Module**: Cleans and prepares text data for analysis.
3. **Translation Module**: Translates text to English using Google Translator for uniform processing.
4. **Domain-Specific Rules Engine**: Applies custom heuristics based on predefined keywords and patterns.
5. **Sentiment Analysis Module**: Utilizes FinBERT for classifying the sentiment of the text.
6. **Classification Logic**: Combines outputs from the rules engine and sentiment analysis to determine lead quality.
7. **Concurrency Module**: Processes texts in parallel for efficiency.
8. **Output Generation**: Provides classified results for download and displays individual text classification.

---

## Detailed Component Breakdown

### 1. User Interface (UI)

- **Purpose**: Facilitates user interaction with the application.
- **Technologies**: Streamlit for web interface, custom CSS for styling.
- **Features**:
    - **File Upload**: Users can upload a CSV file containing utterances.
    - **Individual Text Input**: Allows classification of single text entries.
    - **Progress Indicators**: Displays processing progress during batch classification.
    - **Download Option**: Provides the processed CSV file for download.

**Implementation Highlights**:

- The UI is designed with user experience in mind, featuring clear instructions and an intuitive layout.
- Custom CSS styles are applied to enhance the visual appeal and professionalism of the application.
- Input validation ensures the uploaded file contains the necessary 'utterance' column.

### 2. Data Preprocessing Module

- **Purpose**: Prepares text data for analysis by cleaning and normalizing it.
- **Functions**:
    - **Text Cleaning**: Removes punctuation and extraneous characters.
    - **Lowercasing**: Converts text to lowercase for consistent processing.
    - **Whitespace Stripping**: Eliminates leading and trailing whitespaces.

**Key Function**: `preprocess_text(text)`

- Uses Python's `string` and `str.translate` methods for efficient text cleaning.

### 3. Translation Module

- **Purpose**: Translates text from various languages to English to ensure consistent analysis.
- **Library Used**: `deep_translator.GoogleTranslator`

![alt text](https://media.geeksforgeeks.org/wp-content/uploads/20231226141038/Machine-Translation-Model.png)


**Process**:

- Checks if the text is not empty or whitespace.
- Translates the text from the detected source language to English.
- Handles exceptions to ensure the application remains robust even if translation fails.

### 4. Domain-Specific Rules Engine

- **Purpose**: Applies predefined heuristics to detect specific patterns and keywords indicative of lead quality.
- **Components**:
    - **Keyword Lists**:
        - Positive Synonyms (e.g., 'yes', 'sure', 'absolutely').
        - Negative Synonyms (e.g., 'no', 'never', 'can't pay').
        - Date Keywords (e.g., 'today', 'tomorrow', 'week').
        - Quality Reason Keywords (e.g., 'financial issue', 'medical', 'job loss').
        - Non-Quality Indicators (e.g., 'will not pay', 'not interested').
        - Payment Methods (e.g., 'credit card', 'bank transfer').
        - Greetings and Simple Phrases.
    - **Pattern Matching**:
        - Regular expressions to detect date formats.
        - Functions to check for the presence of keywords in the text.

**Key Functions**:

- `contains_date_or_time(text)`
- `is_greeting(text)`
- `contains_positive_synonym(text)`
- `contains_negative_synonym(text)`
- `contains_quality_reason(text)`
- `contains_non_quality_indicator(text)`
- `contains_payment_intent(text)`
- `contains_future_payment_commitment(text)`

**Logic**:

- The rules engine evaluates the text against various conditions.
- Certain phrases or keywords can directly classify a lead as "Quality" or "Non-Quality" without needing sentiment analysis.

### 5. Sentiment Analysis Module

- **Purpose**: Analyzes the sentiment of the text to aid in lead quality determination.
- **Model Used**: `yiyanghkust/finbert-tone`, a FinBERT **A Large Language Model for Extracting Information from Financial Text.**
- **Libraries**:
    - `transformers.pipeline`
    - `AutoTokenizer`
    - `AutoModelForSequenceClassification`
    
    ![fig-](https://dfzljdn9uc3pi.cloudfront.net/2023/cs-1403/1/fig-2-full.png)
    

**Process**:

- Initializes the tokenizer and model for the FinBERT pipeline.
- Performs sentiment analysis on the preprocessed and translated text.
- Outputs a label: 'POSITIVE', 'NEGATIVE', or 'NEUTRAL'.

**Implementation Details**:

- The pipeline is set with `truncation=True`, `padding=True`, and `max_length=512` to handle long texts efficiently.

### 6. Classification Logic

- **Purpose**: Combines outputs from the domain-specific rules engine and the sentiment analysis to make a final lead quality determination.
- **Process**:
    - **Priority Checks**:
        - Immediate classification if text contains certain keywords (e.g., greetings are classified as "Non-Quality Lead").
    - **Rules Over Sentiment**:
        - Domain-specific rules have precedence over sentiment analysis results.
    - **Sentiment-Based Classification**:
        - If rules are inconclusive, the sentiment label guides the classification.

**Function**: `translate_and_classify(text)`

- Orchestrates the overall classification process for each text entry.
- Handles exceptions and defaults to "Non-Quality Lead" if processing fails.

**Lead Quality Determination**:

- **Quality Lead**:
    - Contains positive synonyms, quality reason keywords, future payment commitments, or positive sentiment.
- **Non-Quality Lead**:
    - Contains negative synonyms, non-quality indicators, or negative sentiment.
- **No Utterance Available**:
    - Empty or whitespace-only text entries.

### 7. Concurrency Module

- **Purpose**: Enhances performance by processing multiple texts simultaneously.
- **Library Used**: `concurrent.futures.ThreadPoolExecutor`
- **Function**: `classify_texts_in_batches(texts, batch_size=100, num_workers=4)`

**Process**:

- Submits classification tasks to a thread pool executor.
- Uses `as_completed` to retrieve results as they become available.
- Updates a progress bar in the UI to reflect processing status.
- Handles exceptions for individual tasks without interrupting the entire process.

**Benefits**:

- Significantly reduces processing time for large datasets.
- Provides a responsive user experience by leveraging multi-threading.

### 8. Output Generation

- **Purpose**: Compiles classification results and presents them to the user.
- **Process**:
    - Adds a new column `lead_quality` to the DataFrame based on classification results.
    - Drops intermediate columns (e.g., `sentiment`) to streamline the output.
    - Saves the processed DataFrame to a CSV file.
- **User Interaction**:
    - Users can download the classified leads using a download button.
    - Individual text classification results are displayed directly in the UI.

---

## User Interface Design and Features

### Streamlit Configuration

- Sets the page title and layout using `st.set_page_config`.
- Applies custom CSS styles to enhance the visual presentation.
- Utilizes Markdown and HTML (with `unsafe_allow_html=True`) for rich text formatting and embedding images.

### UI Components

- **Header and Introduction**:
    - Provides an engaging introduction to the application.
    - Displays a logo and descriptive text to inform users about the application's purpose.
- **File Upload Section**:
    - Allows users to upload a CSV file via `st.file_uploader`.
    - Validates the presence of the 'utterance' column.
    - Displays the dataset's columns for user verification.
- **Progress Indicators**:
    - Uses `st.progress` to visually represent the processing status during classification.
- **Download Button**:
    - Enables users to download the classified leads CSV file.
    - Styled to stand out and encourage user interaction.
- **Individual Text Classification**:
    - Provides a text area for users to input a single text.
    - Displays the classification result upon submission.

### Custom CSS Styling

- Enhances the look and feel of the application by:
    - Customizing fonts, colors, and backgrounds.
    - Styling buttons, text areas, headers, and other UI elements.
    - Ensuring consistency and a professional appearance.

---

## Data Flow and Processing Pipeline

1. **Data Ingestion**:
    - User uploads a CSV file or inputs text manually.
    - The application reads the data into a pandas DataFrame.
2. **Preprocessing**:
    - Text is preprocessed to remove noise and standardize the format.
3. **Translation**:
    - Non-English text is translated to English for uniform analysis.
4. **Domain-Specific Evaluation**:
    - Text is evaluated against domain-specific rules.
    - Immediate classifications are made if certain conditions are met.
5. **Sentiment Analysis**:
    - For texts not conclusively classified by rules, sentiment analysis is performed.
    - The sentiment label contributes to the final classification.
6. **Classification Decision**:
    - The application decides if the lead is "Quality" or "Non-Quality" based on combined insights.
7. **Concurrency Handling**:
    - Multiple texts are processed in parallel to optimize performance.
8. **Result Compilation**:
    - The DataFrame is updated with classification results.
    - Outputs are prepared for user download or display.

---

## Exception Handling and Robustness

- The application is designed to handle exceptions gracefully:
    - **Translation Failures**: Defaults to "Non-Quality Lead" if translation fails.
    - **Empty Inputs**: Recognizes empty or whitespace-only inputs and marks them accordingly.
    - **Processing Errors**: Catches and logs exceptions during classification without stopping the entire process.
- User feedback is provided through status messages, warnings, and error displays.

---

---

## Future Enhancements

- **Model Optimization**: Fine-tuning FinBERT or integrating more advanced models for better accuracy.
- **User Customization**: Allowing users to modify or add domain-specific keywords and rules.
- **Language Support**: Enhancing translation capabilities or incorporating multilingual models to reduce reliance on translation.
- **Analytics Dashboard**: Providing visualizations and statistics on the classification results within the app.

---

## References

- **FinBERT Model**: [yiyanghkust/finbert-tone](https://onlinelibrary.wiley.com/doi/full/10.1111/1911-3846.12832)
- **Streamlit Documentation**: [Streamlit Docs](https://docs.streamlit.io/)
- **Transformers Library**: [Hugging Face Transformers](https://huggingface.co/docs)
- **Deep Translator**: [deep-translator GitHub](https://github.com/nidhaloff/deep-translator)
- **Pandas Library**: [Pandas Documentation](https://pandas.pydata.org/docs/index.html)

---

*This architectural documentation is intended to provide a clear and detailed understanding of the AI-Powered Lead Quality Analyzer's components and their interactions. For any questions or further clarifications, please refer to the source code or reach out to developer.*
