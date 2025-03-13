# Review Analysis Dashboard

This dashboard analyzes app reviews to identify and prioritize user concerns and issues. It uses topic modeling, sentiment analysis, and natural language processing to extract meaningful insights from user reviews.

## Features

- Upload and analyze Excel files containing app reviews
- Automatic sentiment analysis to identify negative and neutral reviews
- Topic modeling to group similar issues and concerns
- Interactive visualization of topic analysis metrics
- Topic interpretation using Mistral AI
- Export functionality for analysis results

## Setup

1. Clone this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```
Note: This will install all required packages including openpyxl for Excel file support.

3. Download NLTK data:
```python
import nltk

# Download required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
```
Note: These downloads will be attempted automatically when you run the app, but you can run them manually if needed.

4. Set up environment variables:
   - Copy `.env.template` to `.env`
   - Add your Hugging Face API key to the `.env` file:
```bash
HUGGINGFACE_API_KEY=your_api_key_here
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run src/app.py
```

2. Open your web browser and navigate to the provided URL (usually http://localhost:8501)

3. Upload your Excel file containing reviews and click "Analyze Reviews"

4. View the analysis results in two tabs:
   - Topic Analysis: Shows the metrics visualization
   - Topic Interpretation: Shows the interpreted results

5. Download the results using the provided buttons

## Input File Format

The Excel file (.xlsx) must contain these required columns:
- content: Text content of the review
- thumbsUpCount: Number of thumbs up/likes for the review (numeric)

The sentiment analysis will be performed automatically on the content, so you don't need to provide sentiment scores.

## Output Files

The dashboard generates two types of output files:
1. Negative & Neutral Reviews: CSV file containing filtered reviews with calculated sentiment
2. Topic Interpretation Results: CSV file containing interpreted topics and their metrics

## Dependencies

See requirements.txt for a complete list of dependencies. Key dependencies include:
- streamlit: For the web interface
- pandas & openpyxl: For Excel file handling
- nltk: For natural language processing and sentiment analysis
- gensim: For topic modeling
- sentence-transformers: For text similarity
- python-dotenv: For environment variable management

## Troubleshooting

If you encounter NLTK resource errors, you can manually download all required resources by running this Python code:
```python
import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
``` 