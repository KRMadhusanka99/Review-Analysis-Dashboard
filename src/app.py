import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from topic_prioritizer import TopicPrioritizer
from topic_interpreter import TopicInterpreter
import os
from dotenv import load_dotenv
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download all required NLTK data
try:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt_tab')
except Exception as e:
    st.warning(f"Some NLTK resources could not be downloaded. This might affect the analysis. Error: {str(e)}")

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Review Analysis Dashboard",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 0.75rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .upload-section {
        margin-bottom: 2rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1.2rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for metrics
if 'app_name' not in st.session_state:
    st.session_state.app_name = "No file uploaded"
if 'total_reviews' not in st.session_state:
    st.session_state.total_reviews = 0
if 'neg_neutral_reviews' not in st.session_state:
    st.session_state.neg_neutral_reviews = 0

# Title and metrics at the top
st.title("Review Analysis Dashboard")

# Display metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("App Name", st.session_state.app_name)
with col2:
    st.metric("Total Reviews", st.session_state.total_reviews)
with col3:
    st.metric("Negative & Neutral Reviews", st.session_state.neg_neutral_reviews)

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Reviews")
    with st.form("upload_form"):
        uploaded_file = st.file_uploader("Select Excel file (.xlsx)", type=['xlsx'])
        submit_button = st.form_submit_button("Analyze Reviews")
    
    # Metric explanations
    st.markdown("### Metric Explanations")
    
    with st.expander("Entropy"):
        st.write("Measures the diversity of keywords in reviews, where lower entropy indicates more focused user concerns that are prioritized higher.")
    
    with st.expander("Topic Prevalence"):
        st.write("Represents how frequently a topic appears across reviews, with higher prevalence indicating a more commonly reported issue.")
    
    with st.expander("Thumbs-Up Count"):
        st.write("Reflects user agreement on the importance of a review, where higher counts indicate greater relevance to users.")
    
    with st.expander("Sentiment Score"):
        st.write("Captures the negativity of reviews using VADER, where more negative sentiment increases the priority of a topic.")

# Main content area
if uploaded_file is not None and submit_button:
    try:
        # Get app name from filename and update session state
        app_name = os.path.splitext(uploaded_file.name)[0]
        st.session_state.app_name = app_name
        
        # Load data
        df = pd.read_excel(uploaded_file)
        st.session_state.total_reviews = len(df)
        
        # Initialize sentiment analyzer
        analyzer = SentimentIntensityAnalyzer()
        
        # Calculate sentiment scores and classify
        df['sentiment_score'] = df['content'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
        df['sentiment'] = df['sentiment_score'].apply(lambda score: 
            'positive' if score >= 0.05 
            else 'negative' if score <= -0.05 
            else 'neutral'
        )
        
        # Update negative and neutral reviews count
        neg_neutral_count = len(df[df['sentiment'].isin(['negative', 'neutral'])])
        st.session_state.neg_neutral_reviews = neg_neutral_count
        
        # Rerun to update metrics
        st.experimental_rerun()
        
        # Initialize topic prioritizer
        prioritizer = TopicPrioritizer()
        
        # Process reviews
        df_processed = prioritizer.process_reviews(df)
        texts = df_processed['processed_content'].tolist()
        thumbs_up_counts = df_processed['thumbsUpCount'].tolist()
        
        # Default topic numbers
        topic_numbers = [5, 7, 8, 10]
        
        # Perform topic modeling and evaluation
        results = prioritizer.evaluate_topics(texts, topic_numbers)
        best_num_topics, best_metrics = max(results.items(), key=lambda x: x[1][1])
        
        # Get topic keywords
        lda_model, dictionary, corpus = prioritizer._perform_topic_modeling(texts, best_num_topics)
        topics = lda_model.show_topics(num_topics=best_num_topics, num_words=10, formatted=False)
        topic_keywords = {f"Topic {topic_num}": [word for word, _ in topic] 
                         for topic_num, topic in enumerate(topics)}
        
        # Calculate metrics
        metrics = prioritizer.calculate_metrics(texts, topic_keywords, thumbs_up_counts, None)
        combined_scores = prioritizer.calculate_combined_scores(metrics)
        
        # Create tabs for visualization and results
        tab1, tab2 = st.tabs(["Topic Analysis", "Topic Interpretation"])
        
        with tab1:
            # Create visualization
            st.subheader("Topic Analysis Scores by Metrics")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            x = range(len(topic_keywords))
            width = 0.2
            
            # Plot bars for each metric
            ax.bar([i - 1.5*width for i in x], metrics['entropy'], width, 
                   label='Entropy', color='skyblue')
            ax.bar([i - 0.5*width for i in x], metrics['prevalence'], width,
                   label='Prevalence', color='lightgreen')
            ax.bar([i + 0.5*width for i in x], metrics['thumbs_up'], width,
                   label='Thumbs Up', color='salmon')
            ax.bar([i + 1.5*width for i in x], metrics['sentiment'], width,
                   label='Sentiment', color='purple')
            
            # Plot combined score line
            ax.plot(x, combined_scores, 'k-', label='Combined Score', linewidth=2)
            ax.plot(x, combined_scores, 'ko')
            
            ax.set_xticks(x)
            ax.set_xticklabels([f'Topic {i+1}' for i in range(len(topic_keywords))], rotation=45)
            ax.set_xlabel('Topics')
            ax.set_ylabel('Scores')
            ax.legend()
            plt.tight_layout()
            
            # Display plot
            st.pyplot(fig)
        
        with tab2:
            # Get API key from environment variable
            api_key = os.getenv('HUGGINGFACE_API_KEY')
            
            if api_key:
                interpreter = TopicInterpreter(api_key)
                
                # Extract relevant sentences and interpret topics
                relevant_sentences = interpreter.extract_relevant_sentences(df_processed, topic_keywords)
                interpretation_results = interpreter.interpret_topics(relevant_sentences, topic_keywords)
                
                # Display interpretation results
                st.subheader("Topic Interpretation Results")
                st.dataframe(interpretation_results)
                
                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    # Save negative and neutral reviews
                    csv = df_processed.to_csv(index=False)
                    st.download_button(
                        label="Download Negative & Neutral Reviews",
                        data=csv,
                        file_name=f"{app_name}_negative_neutral_reviews.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Save interpretation results
                    csv = interpretation_results.to_csv(index=False)
                    st.download_button(
                        label="Download Topic Interpretation Results",
                        data=csv,
                        file_name=f"{app_name}_topic_interpretation.csv",
                        mime="text/csv"
                    )
            else:
                st.error("HUGGINGFACE_API_KEY not found in environment variables. Please add it to your .env file.")
            
    except Exception as e:
        st.error(f"An error occurred while processing the file: {str(e)}")
else:
    if submit_button and not uploaded_file:
        st.sidebar.warning("Please select an Excel file before analyzing.")
