import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from gensim import corpora, models
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords as nltk_stopwords
from nltk.tag import pos_tag
from langdetect import detect
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.coherencemodel import CoherenceModel
from scipy.stats import entropy
from collections import Counter

nltk.download('vader_lexicon')

analyzer = SentimentIntensityAnalyzer()

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# warnings.filterwarnings("ignore")
from google.colab import drive

drive.mount('/content/drive')

# Load Excel dataset
# file_path = '/content/drive/My Drive/Research/MPhil/WebScraping/Journal/Integrating/Data_demand/instagram.xlsx'  # Replace with your file path
file_path = '/content/drive/My Drive/Research/MPhil/WebScraping/Journal/Netflix.xlsx'
df = pd.read_excel(file_path)


# Check for floats in the 'content' column
float_mask = df['content'].apply(lambda x: isinstance(x, (float, int)))

# Remove floats from the 'content' column
df['content'] = df['content'].where(~float_mask, '')

# Remove rows where 'content' is NaN or an empty string
df = df.dropna(subset=['content'])  # Drop NaN values
df = df[df['content'].str.strip() != '']  # Drop empty strings

# Lowercase the 'content' column
df['review'] = df['content'].apply(lambda t: str(t).lower())

# Dictionary of English Contractions
contractions_dict = { "ain't": "are not","'s":" is","aren't": "are not",
                     "can't": "cannot","can't've": "cannot have",
                     "'cause": "because","could've": "could have","couldn't": "could not",
                     "couldn't've": "could not have", "didn't": "did not","doesn't": "does not",
                     "don't": "do not","hadn't": "had not","hadn't've": "had not have",
                     "hasn't": "has not","haven't": "have not","he'd": "he would",
                     "he'd've": "he would have","he'll": "he will", "he'll've": "he will have",
                     "how'd": "how did","how'd'y": "how do you","how'll": "how will",
                     "I'd": "I would", "I'd've": "I would have","I'll": "I will",
                     "I'll've": "I will have","I'm": "I am","I've": "I have", "isn't": "is not",
                     "it'd": "it would","it'd've": "it would have","it'll": "it will",
                     "it'll've": "it will have", "let's": "let us","ma'am": "madam",
                     "mayn't": "may not","might've": "might have","mightn't": "might not",
                     "mightn't've": "might not have","must've": "must have","mustn't": "must not",
                     "mustn't've": "must not have", "needn't": "need not",
                     "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
                     "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
                     "shan't've": "shall not have","she'd": "she would","she'd've": "she would have",
                     "she'll": "she will", "she'll've": "she will have","should've": "should have",
                     "shouldn't": "should not", "shouldn't've": "should not have","so've": "so have",
                     "that'd": "that would","that'd've": "that would have", "there'd": "there would",
                     "there'd've": "there would have", "they'd": "they would",
                     "they'd've": "they would have","they'll": "they will",
                     "they'll've": "they will have", "they're": "they are","they've": "they have",
                     "to've": "to have","wasn't": "was not","we'd": "we would",
                     "we'd've": "we would have","we'll": "we will","we'll've": "we will have",
                     "we're": "we are","we've": "we have", "weren't": "were not","what'll": "what will",
                     "what'll've": "what will have","what're": "what are", "what've": "what have",
                     "when've": "when have","where'd": "where did", "where've": "where have",
                     "who'll": "who will","who'll've": "who will have","who've": "who have",
                     "why've": "why have","will've": "will have","won't": "will not",
                     "won't've": "will not have", "would've": "would have","wouldn't": "would not",
                     "wouldn't've": "would not have","y'all": "you all", "y'all'd": "you all would",
                     "y'all'd've": "you all would have","y'all're": "you all are",
                     "y'all've": "you all have", "you'd": "you would","you'd've": "you would have",
                     "you'll": "you will","you'll've": "you will have", "you're": "you are",
                     "you've": "you have"}

# Regular expression for finding contractions
contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))

# Function for expanding contractions
def expand_contractions(content,contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, content)

# Expanding Contractions in the text data
df.content = df.content.apply(lambda x:expand_contractions(x))

# Remove non-English reviews and short reviews
df = df[df['content'].str.contains(r'\b\w+\b') & (df['content'].str.split().str.len() > 3)]

# Custom stopwords list
custom_stopwords = { 'facebook', 'netflix', 'instagram', 'app', 'application', 'education', 'lecture', 'learn', 'learning', 'online', 'really', 'helpful', 'thanks', 'thank', 'teach', 'student', 'bad', 'great', 'enjoy', 'wonderful', 'love', 'best', 'nice', 'use', 'try', 'easy', 'difficult', 'is', 'are', 'was', 'were', 'be', 'been', 'am', 'have', 'has', 'had', 'do', 'does', 'did', 'may', 'get', 'though', 'could', 'does', 'has', 'have', 'had', 'pp', 'good', 'excellent', 'awesome', 'please', 'they', 'very', 'too', 'like', 'yeah',
'amazing', 'bad', 'ad', 'u', 'say', 'would', 'lovely', 'perfect', 'much', 'yup', 'suck', 'super', 'omg', 'gud', 'yes', 'cool', 'fine', 'hello', 'god', 'alright', 'poor', 'plz', 'pls', 'google', 'facebook', 'not', 'worst', 'anyware', 'everything', 'three', 'ones', 'one', 'two', 'five', 'four', 'old', 'new', 'asap', 'version', 'times', 'update', 'star', 'first', 'rid', 'bit', 'annoying', 'beautiful', 'dear', 'master', 'evernote', 'per', 'line'}

# Combine NLTK and custom stopwords
combined_stopwords = set(nltk_stopwords.words('english')) | custom_stopwords

# Function to remove emojis from text
def remove_emojis(text):
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U0001F004-\U0001F0CF\U00002764\U00002795\U00002B50]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

# Function to map POS tags to WordNet POS tags
def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)

# Preprocessing function (without stopword removal)
def preprocess_text(text):
    if not isinstance(text, str):  # Check if the input is a string
        return ''  # Return an empty string or handle accordingly

    # Remove emojis
    text = remove_emojis(text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # Language detection
    try:
        if detect(text) != 'en':  # Check if the detected language is not English
            return ""  # If it's not English, return an empty string to filter it out
    except:
        return ""  # If an error occurs during language detection, return an empty string

    return text

# Function to remove stopwords
def remove_custom_stopwords(text):
    tokens = word_tokenize(text)
    return ' '.join(word for word in tokens if word.lower() not in combined_stopwords)

# Lemmatization function
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens]
    return ' '.join(lemmatized_tokens)

# Apply preprocessing and lemmatization to the reviews
df['processed_content'] = df['content'].apply(preprocess_text)
df['processed_content'] = df['processed_content'].apply(remove_custom_stopwords)
df['processed_content'] = df['processed_content'].apply(lemmatize_text)

# Convert 'appVersion' column to strings
df['reviewCreatedVersion'] = df['reviewCreatedVersion'].astype(str)

# Discard the third dot-separated component of the app version
df['reviewCreatedVersion'] = df['reviewCreatedVersion'].apply(lambda x: '.'.join(x.split('.')[:2]))

# Convert 'appVersion' column to numeric
df['reviewCreatedVersion'] = pd.to_numeric(df['reviewCreatedVersion'], errors='coerce')

# Get the number of reviews for each app version
review_counts = df['reviewCreatedVersion'].value_counts().sort_index(ascending=False)

# Convert the series to a DataFrame
review_counts_df = review_counts.reset_index()
review_counts_df.columns = ['App Version', 'Number of Reviews']

def get_sentiment_score(text):
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']

df['sentiment_score'] = df['content'].apply(get_sentiment_score)

# Define sentiment classification function
def classify_sentiment(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Apply sentiment classification
df['sentiment'] = df['sentiment_score'].apply(classify_sentiment)

# Filter Negative and Neutral Reviews
df_new = df[(df['sentiment'] == 'negative') | (df['sentiment'] == 'neutral')]

texts = df_new['processed_content'].tolist()
thumbs_up_counts = df_new['thumbsUpCount'].tolist()

# 2. Define LDA topic modeling function
def lda_topic_modeling(texts, num_topics):
    # Tokenize texts
    tokenized_texts = [text.split() for text in texts]

    # Create a dictionary and corpus for the LDA model
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

    # Train the LDA model
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

    return lda_model, dictionary, corpus

# 3. Evaluate topics and find the best number
def evaluate_topics(texts, topic_numbers):
    results = {}
    for num_topics in topic_numbers:
        lda_model, dictionary, corpus = lda_topic_modeling(texts, num_topics)

        # Calculate perplexity
        perplexity = lda_model.log_perplexity(corpus)

        # Calculate coherence score
        coherence_model = CoherenceModel(model=lda_model, texts=[text.split() for text in texts], dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()

        results[num_topics] = (perplexity, coherence_score)

    return results

def find_best_topic_number(results):
    # Find the number of topics with the highest coherence score
    best_num_topics = max(results, key=lambda x: results[x][1])
    return best_num_topics, results[best_num_topics]

def print_results(results):
    print(f"{'Number of Topics':<15} {'Perplexity':<15} {'Coherence Score':<20}")
    print("-" * 50)
    for num_topics, (perplexity, coherence_score) in results.items():
        print(f"{num_topics:<15} {perplexity:<15.2f} {coherence_score:<20.2f}")

# 4. Calculate and print best number of topics
topic_numbers = [5, 7, 8, 10, 12, 14]
results = evaluate_topics(texts, topic_numbers)
print_results(results)

best_num_topics, best_metrics = find_best_topic_number(results)
# print(f"\nBest Number of Topics: {best_num_topics}")
# print(f"Perplexity: {best_metrics[0]:.2f}")
# print(f"Coherence Score: {best_metrics[1]:.2f}")

# 5. Train LDA model with the best number of topics
lda_model, dictionary, corpus = lda_topic_modeling(texts, best_num_topics)

# Print the topics and their keywords
topics = lda_model.show_topics(num_topics=best_num_topics, num_words=10, formatted=False)
topic_keywords = {topic_num: [word for word, _ in topic] for topic_num, topic in topics}
print(f"\nTopics and Keywords (Num Topics = {best_num_topics}):")
for topic_num, keywords in topic_keywords.items():
    print(f"Topic {topic_num}: {', '.join(keywords)}")

# 6. Find synonyms for each keyword using Word2Vec
def find_similar_words(word2vec_model, keywords):
    similar_words = {}
    for word in keywords:
        if word in word2vec_model.wv:
            similar_words[word] = [similar_word for similar_word, _ in word2vec_model.wv.most_similar(word, topn=3)]
    return similar_words

# Expand keywords with synonyms
def expand_keywords_with_synonyms(topic_keywords, word2vec_model):
    expanded_keywords = {}
    for topic_num, keywords in topic_keywords.items():
        keyword_set = set(keywords)
        synonyms = find_similar_words(word2vec_model, keywords)
        expanded_keyword_set = keyword_set.copy()

        # Add synonyms to the expanded keyword set
        for keyword, similar_words in synonyms.items():
            expanded_keyword_set.update(similar_words)

        expanded_keywords[topic_num] = expanded_keyword_set
    return expanded_keywords

# Train or load Word2Vec model
word2vec_model = Word2Vec(sentences=[text.split() for text in texts], vector_size=100, window=5, min_count=1, workers=4)

# Create the expanded keywords dictionary
expanded_keywords = expand_keywords_with_synonyms(topic_keywords, word2vec_model)

# To visualize the expanded keywords, you may need to prepare new topics for visualization:
expanded_topics = [(num, [(word, 1) for word in keywords]) for num, keywords in expanded_keywords.items()]

def calculate_entropy(texts, expanded_keywords):
    entropy_scores = {}
    for topic_num, keywords in expanded_keywords.items():
        keyword_set = set(keywords)
        keyword_counts = Counter()
        for text in texts:
            words = text.split()
            keyword_counts.update(word for word in words if word in keyword_set)
        total_count = sum(keyword_counts.values())
        keyword_probs = [count / total_count for count in keyword_counts.values()]
        entropy_scores[topic_num] = entropy(keyword_probs)
    return entropy_scores

def calculate_prevalence(texts, expanded_keywords):
    prevalence_scores = {}
    total_texts = len(texts)
    for topic_num, keywords in expanded_keywords.items():
        keyword_set = set(keywords)
        count = sum(any(word in keyword_set for word in text.split()) for text in texts)
        prevalence_scores[topic_num] = count / total_texts
    return prevalence_scores

def calculate_thumbs_up_count(texts, thumbs_up_counts, topic_keywords, word2vec_model):
    thumbs_up_scores = {}
    for topic_num, keywords in topic_keywords.items():
        keyword_set = set(keywords)
        total_thumbs_up = 0
        for text, thumbs_up in zip(texts, thumbs_up_counts):
            words = set(text.split())
            expanded_keywords = set()
            for keyword in keyword_set:
                expanded_keywords.add(keyword)
                if keyword in word2vec_model.wv:
                    expanded_keywords.update([similar_word for similar_word, _ in word2vec_model.wv.most_similar(keyword, topn=3)])
            # Check if at least three keywords or their synonyms are present
            if len(expanded_keywords & words) >= 3:
                total_thumbs_up += thumbs_up
        thumbs_up_scores[topic_num] = total_thumbs_up
    return thumbs_up_scores

def calculate_sentiment_scores(texts, sentiment_analyzer, topic_keywords, word2vec_model):
    sentiment_scores = {}
    for topic_num, keywords in topic_keywords.items():
        keyword_set = set(keywords)
        expanded_keywords = set(keyword_set)  # Start with the original keywords
        # Find synonyms for each keyword and add to expanded_keywords
        for keyword in keyword_set:
            if keyword in word2vec_model.wv:
                expanded_keywords.update([similar_word for similar_word, _ in word2vec_model.wv.most_similar(keyword, topn=3)])

        scores = []
        for text in texts:
            words = set(text.split())
            # Check if at least three keywords or their synonyms are present
            if len(expanded_keywords & words) >= 3:
                score = sentiment_analyzer.polarity_scores(text)['compound']
                scores.append(score)

        sentiment_scores[topic_num] = np.mean(scores) if scores else 0

    return sentiment_scores

# Normalize function to ensure values are between 0 and 1
def normalize(values):
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [0.5] * len(values)  # Neutral value if no variation
    return [(val - min_val) / (max_val - min_val) for val in values]

# Get the expanded content for entropy, prevalence, sentiment, and thumbs up calculation
expanded_texts = df_new['processed_content'].tolist()

# 1. Calculate metrics
entropy_scores = calculate_entropy(expanded_texts, topic_keywords)
prevalence_scores = calculate_prevalence(expanded_texts, topic_keywords)
thumbs_up_scores = calculate_thumbs_up_count(expanded_texts, thumbs_up_counts, topic_keywords, word2vec_model)
sentiment_scores = calculate_sentiment_scores(expanded_texts, analyzer, topic_keywords, word2vec_model)

# 2. Invert and normalize entropy scores
min_entropy = min(entropy_scores.values())
max_entropy = max(entropy_scores.values())
inverted_entropy_scores = {topic_num: 1 - (score - min_entropy) / (max_entropy - min_entropy) for topic_num, score in entropy_scores.items()}
normalized_entropy = normalize(list(inverted_entropy_scores.values()))

# 3. Normalize sentiment scores directly
inverted_sentiment_scores = {topic_num: -score for topic_num, score in sentiment_scores.items()}
normalized_sentiment = normalize(list(inverted_sentiment_scores.values()))

# 4. Normalize other metrics
normalized_prevalence = list(prevalence_scores.values())  # Already in [0, 1] as they are proportions
normalized_thumbs_up = normalize(list(thumbs_up_scores.values()))

# Combine scores with equal weights (0.25 for each component)
combined_scores = [
    0.25 * n_entropy + 0.25 * n_prevalence + 0.25 * n_thumbs_up + 0.25 * n_sentiment
    for n_entropy, n_prevalence, n_thumbs_up, n_sentiment in zip(
        normalized_entropy, normalized_prevalence, normalized_thumbs_up, normalized_sentiment
    )
]

# Combine scores with equal weights (0.25 for each component)
Heuristic_combined_scores = [
    0.25 * n_entropy + 0.3 * n_prevalence + 0.25 * n_thumbs_up + 0.2 * n_sentiment
    for n_entropy, n_prevalence, n_thumbs_up, n_sentiment in zip(
        normalized_entropy, normalized_prevalence, normalized_thumbs_up, normalized_sentiment
    )
]

# Step 4: Create a DataFrame to display results
results_df = pd.DataFrame({
    'Topic': [f'Topic {i+1}' for i in range(len(topic_keywords))],
    'Keywords': [', '.join(keywords[:3]) for keywords in topic_keywords.values()],  # First 3 keywords
    'Entropy': normalized_entropy,
    'Prevalence': normalized_prevalence,
    'Thumbs Up': normalized_thumbs_up,
    'Sentiment': normalized_sentiment,
    'Combined Score': combined_scores,
    'Heuristic_Combined Score': Heuristic_combined_scores
})

# Step 3: Sort by combined scores in descending order
results_df = results_df.sort_values(by='Combined Score', ascending=False)

# topic_keywords_list = results_df["Keywords"].tolist()

# Display the results_df
print("\nMetrics for Each Topic:")
results_df.head()

def find_example_reviews(df, topic_keywords, word2vec_model, num_examples=3):
    example_reviews = {}
    for topic_num, keywords in topic_keywords.items():
        keyword_set = set(keywords)
        synonyms = set()
        find_similar_words(word2vec_model, topic_keywords)

        topic_reviews = []
        for _, row in df.iterrows():
            text = row['processed_content']
            words = set(text.split())
            if len(keyword_set & words) >= 3:
                if row['sentiment'] < 0:  # Ensure negative sentiment
                    topic_reviews.append({
                        'review': row['content'],
                        'text': text,
                        'sentiment': row['sentiment']
                    })
            if len(topic_reviews) >= num_examples:
                break
        example_reviews[topic_num] = topic_reviews
    return example_reviews

# Filter negative reviews
df_new['sentiment'] = df_new['processed_content'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
negative_reviews_df = df_new[df_new['sentiment'] < 0]

# Get example reviews
example_reviews = find_example_reviews(negative_reviews_df, topic_keywords, word2vec_model)

# Print example reviews for each topic
print("Example Reviews for Each Topic:")
for topic_num, reviews in example_reviews.items():
    # Fetch the correct keywords for the current topic and convert to a list
    keywords = list(topic_keywords[topic_num])
    print(f"\nTopic {topic_num} (Keywords: {', '.join(keywords[:3])}):")
    for review in reviews:
        print(f"Review: {review['review']}")
        print(f"Sentiment Score: {review['sentiment']}")
        print("-" * 80)
results_df.head(15)

# Print entropy, prevalence, thumbs up, sentiment, and combined scores
print("\nTopic Analysis Results:")
print(f"{'Topic':<10} {'Keywords':<50} {'Entropy':<10} {'Prevalence':<12} {'Thumbs Up':<10} {'Sentiment':<12} {'Combined Score':<15}")
print("-" * 110)
for topic_num, (keywords, entropy, prevalence, thumbs_up, sentiment, combined) in enumerate(
    zip(topic_keywords.values(), normalized_entropy, normalized_prevalence, normalized_thumbs_up, normalized_sentiment, combined_scores)):
    keywords_str = ', '.join(keywords[:10])  # Display only the first 10 keywords for brevity
    print(f"{topic_num + 1:<10} {keywords_str:<50} {entropy:<10.2f} {prevalence:<12.2f} {thumbs_up:<10.2f} {sentiment:<12.2f} {combined:<15.2f}")

# Print the best number of topics and their keywords
print(f"\nBest Number of Topics: {best_num_topics}")
for topic, keywords in topic_keywords.items():
    print(f"Topic {topic}: {', '.join(keywords)}")

# Data preparation
topic_labels = [f"Topic {i+1}" for i in range(len(topic_keywords))]
keywords_labels = [', '.join(keywords[:3]) for keywords in topic_keywords.values()]  # Shortened keyword display
combined_scores = np.array(combined_scores)

# Set a consistent color palette
colors = plt.get_cmap('tab20c')(np.linspace(0, 1, 4))

# Plotting
fig, ax = plt.subplots(figsize=(12, 8))

bar_width = 0.2
index = np.arange(len(topic_keywords))

# Plot bars for each metric with different colors
ax.bar(index, normalized_entropy, bar_width, color=colors[0], label='Entropy')
ax.bar(index + bar_width, normalized_prevalence, bar_width, color=colors[1], label='Prevalence')
ax.bar(index + 2 * bar_width, normalized_thumbs_up, bar_width, color=colors[2], label='Thumbs Up')
ax.bar(index + 3 * bar_width, normalized_sentiment, bar_width, color=colors[3], label='Sentiment')

# Combined score line plot
ax.plot(index + 1.5 * bar_width, combined_scores, color='black', marker='o', linestyle='-', linewidth=2, label='Combined Score')

# Labels and titles
ax.set_xlabel('Topics')
ax.set_ylabel('Normalized Scores')
ax.set_title('Topic Analysis Scores by Metrics')

# Set x-ticks and labels
ax.set_xticks(index + 1.5 * bar_width)
ax.set_xticklabels(topic_labels, rotation=45, ha="right")

# Add topic keywords to legend
handles, labels = ax.get_legend_handles_labels()
for i, (label, keywords) in enumerate(zip(topic_labels, keywords_labels)):
    handles.append(plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2))
    labels.append(f"{label} Keywords: {keywords}")

# Place legend outside the plot
ax.legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(1, 1))

# Add grid and layout adjustment
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout(rect=[0, 0, 0.75, 1])  # Adjust plot area to fit legend

# Show the plot
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Create a DataFrame for heatmap
data = {
    'Topic': topic_labels,
    'Entropy': normalized_entropy,
    'Prevalence': normalized_prevalence,
    'Thumbs Up': normalized_thumbs_up,
    'Sentiment': normalized_sentiment
}
df_heatmap = pd.DataFrame(data).set_index('Topic')

# Plotting
plt.figure(figsize=(14, 8))
sns.heatmap(df_heatmap, annot=True, cmap='YlGnBu', fmt='.2f', linewidths=.5)
plt.title('Heatmap of Topic Analysis Scores')
plt.show()

# # Function to extract the top 10 most relevant sentences for each topic based on thumbsUpCount
# def extract_relevant_sentences(df_new, topic_keywords_list):
#     results = {}

#     for topic, keywords in topic_keywords_list.items():
#         relevant_sentences = []  # Store tuples of (sentence, thumbsUpCount) for each relevant sentence

#         for index, row in df.iterrows():
#             review_text = row['content']
#             thumbs_up = row['thumbsUpCount']  # Get the thumbs-up count for the review

#             # Check if review_text is a string before proceeding
#             if isinstance(review_text, str):
#                 review_text = review_text.lower()  # Convert to lowercase
#                 sentences = re.split(r'(?<=[.!?]) +', review_text)  # Split into sentences

#                 for sentence in sentences:
#                     keywords_found = [keyword for keyword in keywords if keyword in sentence]

#                     # Check if the sentence contains at least 5 unique keywords
#                     if len(set(keywords_found)) >= 5:
#                         # Append sentence with thumbs-up count
#                         relevant_sentences.append((sentence.strip(), thumbs_up))

#         # Sort sentences by thumbs-up count in descending order and keep only the top 15
#         top_sentences = sorted(relevant_sentences, key=lambda x: x[1], reverse=True)[:15]

#         # Store only the sentences in the results dictionary
#         results[topic] = [sentence for sentence, thumbs_up in top_sentences]

#     return results

# relevant_sentences = extract_relevant_sentences(df, topic_keywords_list)

# # Display results
# for topic, sentences in relevant_sentences.items():
#     print(f"\n{topic} - Top 15 Relevant Sentences:")
#     for sentence in sentences:
#         print(f"- {sentence}")

# # Extract relevant sentences
# relevant_sentences = extract_relevant_sentences(df, topic_keywords_list)

# import requests

# # Function to get response from Hugging Face's Mistral AI model
# def get_mistral_response(api_key, prompt):
#     url = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
#     headers = {
#         "Authorization": f"Bearer {api_key}",
#         "Content-Type": "application/json"
#     }
#     data = {
#         "inputs": prompt,
#         "parameters": {
#             "max_new_tokens": 180,
#             "temperature": 0.3
#         }
#     }

#     response = requests.post(url, headers=headers, json=data)

#     # Check if the response is valid and parse JSON correctly
#     if response.status_code == 200:
#         generated_text = response.json()[0].get("generated_text", "").strip()
#         return generated_text.split("\n", 1)[-1].strip() if "\n" in generated_text else generated_text
#     else:
#         print(f"Error: {response.status_code}, {response.text}")
#         return None

# # Your Hugging Face API key
# api_key = "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"
# # Initialize a list to store each topic's data for the DataFrame
# data = []

# # Prepare prompts for each topic
# for topic, sentences in relevant_sentences.items():
#     if sentences:  # Proceed only if there are relevant sentences
#         # Join relevant sentences to form a single string
#         joined_sentences = " ".join(sentences)

#         # User Concerns Prompt
#         concern_prompt = f"Identify and summarize the issues, problems or frustrations which users experienced, and identify corresponding app features and functionalities in these user reviews: {joined_sentences[:15]} relevant to the keywords: {', '.join(topic_keywords[topic])}."

#         # Get responses
#         user_concerns = get_mistral_response(api_key, concern_prompt)

#         # Print results
#         print(f"\n{topic} - User Concerns:")
#         if user_concerns:
#             print(user_concerns)
#         else:
#             print("No user concerns found.")

#              # Store results in the list for DataFrame
#         data.append({
#             "Topic": topic,
#             "Topic Keywords": ', '.join(topic_keywords_list[topic]),
#             "Generated User Concerns": user_concerns if user_concerns else "No user concerns found."
#         })

# # Convert the list of dictionaries into a DataFrame
# df_results1 = pd.DataFrame(data)

# # Print or save the DataFrame
# df_results1.head(10)

# from sentence_transformers import SentenceTransformer, util
# from textstat import flesch_reading_ease

# # Load pre-trained model for embeddings
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Function to calculate cosine similarity
# def calculate_relevance(keywords, generated_text):
#     keywords_embedding = model.encode(" ".join(keywords), convert_to_tensor=True)
#     text_embedding = model.encode(generated_text, convert_to_tensor=True)
#     relevance_score = util.cos_sim(keywords_embedding, text_embedding).item()
#     return relevance_score

# # Normalize clarity score to range [0, 1]
# def normalize_clarity(clarity_score, max_clarity=100):
#     return min(clarity_score / max_clarity, 1.0)

# # Composite score calculation
# def calculate_composite_score(relevance, clarity, relevance_weight=0.7, clarity_weight=0.3, max_clarity=100):
#     normalized_clarity = normalize_clarity(clarity, max_clarity)
#     return (relevance * relevance_weight) + (normalized_clarity * clarity_weight)

# # Add relevance, clarity, and combined scores to the DataFrame
# scored_data = []
# for index, row in df_results1.iterrows():
#     topic_keywords_list = row['Topic Keywords'].split(", ")
#     generated_text = row['Generated User Concerns']

#     if generated_text != "No user concerns found.":
#         relevance = calculate_relevance(topic_keywords_list, generated_text)
#         clarity = flesch_reading_ease(generated_text)
#         normalized_clarity = normalize_clarity(clarity)
#         composite_score = calculate_composite_score(relevance, clarity)

#         scored_data.append({
#             "Relevance Score": relevance,
#             "Normalized Clarity Score": normalized_clarity,
#             "Combined Score": composite_score
#         })
#     else:
#         scored_data.append({
#             "Relevance Score": 0.0,
#             "Normalized Clarity Score": 0.0,
#             "Combined Score": 0.0
#         })

# # Add new columns to the existing DataFrame
# scores_df = pd.DataFrame(scored_data)
# df_results1 = pd.concat([df_results1, scores_df], axis=1)

# df_results1.head()

# average_combined_score = df_results1['Combined Score'].mean()
# print(f"Average Combined Score: {average_combined_score}")

# df_results1.iloc[2,2]

# # Save the relevant reviews to a new CSV file
# df_results1.to_excel('/content/drive/My Drive/Research/MPhil/WebScraping/Journal/Integrating/Topic interpretation/temperature 0.4/test.xlsx', index=False)

