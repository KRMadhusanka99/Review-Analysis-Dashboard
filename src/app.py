import streamlit as st
# Set page configuration must be the first st command
st.set_page_config(
    page_title="Review Analysis Dashboard",
    layout="wide"
)

import pandas as pd
import matplotlib.pyplot as plt
from topic_prioritizer import TopicPrioritizer
import nltk
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize error flag for SentenceTransformer
st.session_state['transformer_error'] = False

# Safely import TopicInterpreter
try:
    from topic_interpreter import TopicInterpreter
    TOPIC_INTERPRETER_AVAILABLE = True
except Exception as e:
    TOPIC_INTERPRETER_AVAILABLE = False
    st.warning("Topic Interpreter functionality is not available. Please ensure you have the correct versions of torch and transformers installed.")

# Download required NLTK data at startup
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
    except Exception as e:
        st.warning(f"Some NLTK resources could not be downloaded. Error: {str(e)}")

# Call the download function
download_nltk_data()

# Add custom CSS
st.markdown("""
    <style>
    .main { padding: 1rem; }
    .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    div[data-testid="stVerticalBlock"] { gap: 0.5rem; }
    .metric-card {
        background-color: #f8f9fa;
        padding: 0.75rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .stButton>button { width: 100%; margin-top: 1rem; }
    .upload-section { margin-bottom: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 2rem; }
    div[data-testid="stMetricLabel"] { font-size: 1.2rem; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'app_name' not in st.session_state:
    st.session_state['app_name'] = "No file uploaded"
if 'total_reviews' not in st.session_state:
    st.session_state['total_reviews'] = 0
if 'neg_neutral_reviews' not in st.session_state:
    st.session_state['neg_neutral_reviews'] = 0
if 'processed_data' not in st.session_state:
    st.session_state['processed_data'] = None
if 'topic_analysis' not in st.session_state:
    st.session_state['topic_analysis'] = None
if 'file_processed' not in st.session_state:
    st.session_state['file_processed'] = False
if 'current_df' not in st.session_state:
    st.session_state['current_df'] = None
if 'current_interpretation' not in st.session_state:
    st.session_state['current_interpretation'] = None

# Title
st.title("Review Analysis Dashboard")

# Sidebar
with st.sidebar:
    st.header("Upload Reviews")
    uploaded_file = st.file_uploader("Select Excel file (.xlsx)", type=['xlsx'])
    
    # Update app name immediately when file is uploaded
    if uploaded_file is not None:
        app_name = os.path.splitext(uploaded_file.name)[0]
        st.session_state['app_name'] = app_name
    
    analyze_button = st.button("Analyze Reviews")
    
    st.markdown("### Metric Explanations")
    with st.expander("Entropy"):
        st.write("Measures the diversity of keywords in reviews, where lower entropy indicates more focused user concerns.")
    with st.expander("Topic Prevalence"):
        st.write("Represents how frequently a topic appears across reviews, with higher prevalence indicating more common issues.")
    with st.expander("Thumbs-Up Count"):
        st.write("Reflects user agreement on review importance, where higher counts indicate greater relevance.")
    with st.expander("Sentiment Score"):
        st.write("Captures the negativity of reviews, where more negative sentiment increases topic priority.")

# Display metrics
metrics_placeholder = st.empty()
def update_metrics():
    col1, col2, col3 = metrics_placeholder.columns(3)
    with col1:
        st.metric("App Name", st.session_state['app_name'])
    with col2:
        st.metric("Total Reviews", st.session_state['total_reviews'])
    with col3:
        st.metric("Negative & Neutral Reviews", st.session_state['neg_neutral_reviews'])

update_metrics()

# Cache the data processing function
@st.cache_data
def process_data(df):
    prioritizer = TopicPrioritizer()
    return prioritizer.process_reviews(df)

# Cache the topic modeling function
@st.cache_data
def perform_topic_analysis(texts, thumbs_up_counts, num_topics):
    prioritizer = TopicPrioritizer()
    
    # Perform topic modeling
    lda_model, dictionary, corpus = prioritizer._perform_topic_modeling(texts, num_topics)
    
    # Get topics in the correct format
    topics = []
    for i in range(num_topics):
        topic_terms = lda_model.show_topic(i, 10)  # Get top 10 words for each topic
        topics.append(topic_terms)
    
    # Create topic keywords dictionary
    topic_keywords = {f"Topic {i}": [word for word, _ in topics[i]] 
                     for i in range(num_topics)}
    
    # Calculate metrics and combined scores
    metrics = prioritizer.calculate_metrics(texts, topic_keywords, thumbs_up_counts)
    combined_scores = prioritizer.calculate_combined_scores(metrics)
    
    return topic_keywords, metrics, combined_scores

# Main content area
if uploaded_file is not None and analyze_button:
    try:
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Read the Excel file
        status_text.text('Loading data...')
        df = pd.read_excel(uploaded_file)
        st.session_state['total_reviews'] = len(df)
        progress_bar.progress(20)
        
        # Process reviews
        status_text.text('Processing reviews...')
        df_processed = process_data(df)
        st.session_state['neg_neutral_reviews'] = len(df_processed)
        st.session_state['processed_data'] = df_processed
        progress_bar.progress(40)
        
        # Update metrics display after processing
        update_metrics()
        
        if len(df_processed) > 0:
            # Prepare data for topic analysis
            status_text.text('Analyzing topics...')
            texts = df_processed['processed_content'].tolist()
            thumbs_up_counts = df_processed['thumbsUpCount'].tolist()
            
            # Perform topic analysis with fixed number of topics (8)
            topic_keywords, metrics, combined_scores = perform_topic_analysis(
                texts, thumbs_up_counts, num_topics=8
            )
            progress_bar.progress(70)
            
            # Store analysis results
            st.session_state['topic_analysis'] = {
                'topic_keywords': topic_keywords,
                'metrics': metrics,
                'combined_scores': combined_scores
            }
            
            # Create tabs for visualization
            status_text.text('Generating visualizations...')
            if TOPIC_INTERPRETER_AVAILABLE:
                tab1, tab2, tab3 = st.tabs(["Topic Analysis", "Topic Interpretation", "Review Details"])
            else:
                tab1, tab3 = st.tabs(["Topic Analysis", "Review Details"])
            
            with tab1:
                st.subheader("Topic Analysis Results")
                
                # Generate and display plot
                prioritizer = TopicPrioritizer()
                fig = prioritizer.generate_topic_analysis_plot(
                    topic_keywords, metrics, combined_scores
                )
                st.pyplot(fig)
                plt.close(fig)
                
                # Display topic keywords
                st.subheader("Topic Keywords")
                for topic_num, keywords in topic_keywords.items():
                    st.write(f"{topic_num}: {', '.join(keywords[:10])}")
            
            if TOPIC_INTERPRETER_AVAILABLE:
                with tab2:
                    st.subheader("Topic Interpretation Results")
                    
                    # Get API key from environment variable
                    api_key = os.getenv('HUGGINGFACE_API_KEY')
                    
                    if api_key:
                        try:
                            # Initialize topic interpreter with error handling
                            try:
                                interpreter = TopicInterpreter(api_key)
                                
                                # Extract relevant sentences and interpret topics
                                status_text.text('Interpreting topics...')
                                relevant_sentences = interpreter.extract_relevant_sentences(df_processed, topic_keywords)
                                interpretation_results = interpreter.interpret_topics(relevant_sentences, topic_keywords)
                                
                                # Store interpretation results in session state
                                st.session_state['current_interpretation'] = interpretation_results
                                
                                # Display interpretation results
                                st.dataframe(interpretation_results, use_container_width=True)
                                
                                # Add download button for interpretation results
                                csv = interpretation_results.to_csv(index=False)
                                st.download_button(
                                    label="Download Interpretation Results",
                                    data=csv,
                                    file_name=f"{app_name}_topic_interpretation.csv",
                                    mime="text/csv",
                                    key='download_interpretation'
                                )
                            except Exception as transformer_error:
                                st.error(f"Error initializing Topic Interpreter: {str(transformer_error)}")
                                st.info("Please ensure you have the correct versions of the following packages installed:")
                                st.code("""
                                    pip install torch==2.1.0
                                    pip install sentence-transformers==2.2.2
                                    pip install transformers==4.35.2
                                """)
                        except Exception as e:
                            st.error(f"Error in topic interpretation: {str(e)}")
                    else:
                        st.error("HUGGINGFACE_API_KEY not found in environment variables. Please add it to your .env file.")
            
            with tab3:
                st.subheader("Negative & Neutral Reviews")
                
                # Store review data in session state
                review_df = df_processed[['content', 'sentiment', 'thumbsUpCount']].copy()
                review_df.columns = ['Review', 'Sentiment', 'Thumbs Up']
                st.session_state['current_df'] = review_df
                
                # Display review table
                st.dataframe(review_df, use_container_width=True)
                
                # Add download button
                csv = review_df.to_csv(index=False)
                st.download_button(
                    label="Download Reviews",
                    data=csv,
                    file_name=f"{app_name}_negative_neutral_reviews.csv",
                    mime="text/csv",
                    key='download_reviews'
                )
            
            progress_bar.progress(100)
            status_text.text('Analysis complete!')
            st.session_state['file_processed'] = True
            
        else:
            st.warning("No negative or neutral reviews found in the dataset.")
            
    except Exception as e:
        st.error(f"An error occurred while processing the file: {str(e)}")
        st.exception(e)
else:
    # Display saved results if they exist
    if st.session_state['file_processed'] and st.session_state['topic_analysis'] is not None:
        if TOPIC_INTERPRETER_AVAILABLE:
            tab1, tab2, tab3 = st.tabs(["Topic Analysis", "Topic Interpretation", "Review Details"])
        else:
            tab1, tab3 = st.tabs(["Topic Analysis", "Review Details"])
        
        with tab1:
            st.subheader("Topic Analysis Results")
            
            # Regenerate plot from saved analysis
            prioritizer = TopicPrioritizer()
            topic_analysis = st.session_state['topic_analysis']
            fig = prioritizer.generate_topic_analysis_plot(
                topic_analysis['topic_keywords'],
                topic_analysis['metrics'],
                topic_analysis['combined_scores']
            )
            st.pyplot(fig)
            plt.close(fig)
            
            # Display topic keywords
            st.subheader("Topic Keywords")
            for topic_num, keywords in topic_analysis['topic_keywords'].items():
                st.write(f"{topic_num}: {', '.join(keywords[:10])}")
        
        if TOPIC_INTERPRETER_AVAILABLE:
            with tab2:
                st.subheader("Topic Interpretation Results")
                if st.session_state['current_interpretation'] is not None:
                    st.dataframe(st.session_state['current_interpretation'], use_container_width=True)
                    
                    # Add download button
                    csv = st.session_state['current_interpretation'].to_csv(index=False)
                    st.download_button(
                        label="Download Interpretation Results",
                        data=csv,
                        file_name=f"{st.session_state['app_name']}_topic_interpretation.csv",
                        mime="text/csv",
                        key='download_interpretation'
                    )
        
        with tab3:
            st.subheader("Negative & Neutral Reviews")
            if st.session_state['current_df'] is not None:
                st.dataframe(st.session_state['current_df'], use_container_width=True)
                
                # Add download button
                csv = st.session_state['current_df'].to_csv(index=False)
                st.download_button(
                    label="Download Reviews",
                    data=csv,
                    file_name=f"{st.session_state['app_name']}_negative_neutral_reviews.csv",
                    mime="text/csv",
                    key='download_reviews'
                )
    
    if analyze_button and not uploaded_file:
        st.warning("Please select an Excel file before analyzing.")
