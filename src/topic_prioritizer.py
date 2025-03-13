import pandas as pd
import numpy as np
from gensim import corpora, models
from gensim.models import Word2Vec
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.stats import entropy
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langdetect import detect
import re

# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Custom stopwords
custom_stopwords = {
    'facebook', 'netflix', 'instagram', 'app', 'application', 'education', 
    'lecture', 'learn', 'learning', 'online', 'really', 'helpful', 'thanks', 
    'thank', 'teach', 'student', 'bad', 'great', 'enjoy', 'wonderful', 'love', 
    'best', 'nice', 'use', 'try', 'easy', 'difficult', 'is', 'are', 'was', 
    'were', 'be', 'been', 'am', 'have', 'has', 'had', 'do', 'does', 'did', 
    'may', 'get', 'though', 'could', 'does', 'has', 'have', 'had', 'pp', 
    'good', 'excellent', 'awesome', 'please', 'they', 'very', 'too', 'like', 
    'yeah', 'amazing', 'bad', 'ad', 'u', 'say', 'would', 'lovely', 'perfect', 
    'much', 'yup', 'suck', 'super', 'omg', 'gud', 'yes', 'cool', 'fine', 
    'hello', 'god', 'alright', 'poor', 'plz', 'pls', 'google', 'facebook', 
    'not', 'worst', 'anyware', 'everything', 'three', 'ones', 'one', 'two', 
    'five', 'four', 'old', 'new', 'asap', 'version', 'times', 'update', 
    'star', 'first', 'rid', 'bit', 'annoying', 'beautiful', 'dear', 'master', 
    'evernote', 'per', 'line'
}

class TopicPrioritizer:
    def __init__(self):
        self.combined_stopwords = set(stopwords.words('english')) | custom_stopwords
        self.lemmatizer = WordNetLemmatizer()
        self.analyzer = SentimentIntensityAnalyzer()

    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ''
        
        # Remove emojis
        text = self._remove_emojis(text)
        
        # Remove punctuation and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove non-ASCII characters
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        
        # Language detection
        try:
            if detect(text) != 'en':
                return ""
        except:
            return ""
            
        return text

    def _remove_emojis(self, text):
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
            "\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF"
            "\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF"
            "\U0001F004-\U0001F0CF\U00002764\U00002795\U00002B50]+",
            flags=re.UNICODE
        )
        return emoji_pattern.sub(r'', text)

    def remove_stopwords(self, text):
        tokens = word_tokenize(text)
        return ' '.join(word for word in tokens if word.lower() not in self.combined_stopwords)

    def lemmatize_text(self, text):
        tokens = word_tokenize(text)
        return ' '.join(self.lemmatizer.lemmatize(word) for word in tokens)

    def get_sentiment_score(self, text):
        return self.analyzer.polarity_scores(text)['compound']

    def process_reviews(self, df):
        df['processed_content'] = df['content'].apply(self.preprocess_text)
        df['processed_content'] = df['processed_content'].apply(self.remove_stopwords)
        df['processed_content'] = df['processed_content'].apply(self.lemmatize_text)
        df['sentiment_score'] = df['content'].apply(self.get_sentiment_score)
        df['sentiment'] = df['sentiment_score'].apply(self._classify_sentiment)
        return df[df['sentiment'].isin(['negative', 'neutral'])]

    def _classify_sentiment(self, score):
        if score >= 0.05:
            return 'positive'
        elif score <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    def perform_topic_modeling(self, texts, num_topics):
        tokenized_texts = [text.split() for text in texts]
        dictionary = corpora.Dictionary(tokenized_texts)
        corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
        lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
        return lda_model, dictionary, corpus

    def evaluate_topics(self, texts, topic_numbers):
        results = {}
        for num_topics in topic_numbers:
            lda_model, dictionary, corpus = self.perform_topic_modeling(texts, num_topics)
            perplexity = lda_model.log_perplexity(corpus)
            coherence_model = models.CoherenceModel(
                model=lda_model, 
                texts=[text.split() for text in texts], 
                dictionary=dictionary, 
                coherence='c_v'
            )
            coherence_score = coherence_model.get_coherence()
            results[num_topics] = (perplexity, coherence_score)
        return results

    def calculate_metrics(self, texts, topic_keywords, thumbs_up_counts, word2vec_model):
        entropy_scores = self._calculate_entropy(texts, topic_keywords)
        prevalence_scores = self._calculate_prevalence(texts, topic_keywords)
        thumbs_up_scores = self._calculate_thumbs_up(texts, thumbs_up_counts, topic_keywords, word2vec_model)
        sentiment_scores = self._calculate_sentiment(texts, topic_keywords, word2vec_model)
        
        return {
            'entropy': self._normalize(list(entropy_scores)),
            'prevalence': list(prevalence_scores),
            'thumbs_up': self._normalize(list(thumbs_up_scores)),
            'sentiment': self._normalize(list(sentiment_scores))
        }

    def _calculate_entropy(self, texts, topic_keywords):
        entropy_scores = []
        for keywords in topic_keywords.values():
            keyword_counts = Counter()
            for text in texts:
                words = text.split()
                keyword_counts.update(word for word in words if word in keywords)
            
            total_count = sum(keyword_counts.values())
            if total_count > 0:
                probs = [count/total_count for count in keyword_counts.values()]
                entropy_scores.append(1 - entropy(probs))  # Invert entropy
            else:
                entropy_scores.append(0)
        
        return self._normalize(entropy_scores)

    def _calculate_prevalence(self, texts, topic_keywords):
        prevalence_scores = []
        total_texts = len(texts)
        
        for keywords in topic_keywords.values():
            count = sum(1 for text in texts if any(word in text for word in keywords))
            prevalence_scores.append(count / total_texts if total_texts > 0 else 0)
        
        return prevalence_scores  # Already normalized

    def _calculate_thumbs_up(self, texts, thumbs_up_counts, topic_keywords, word2vec_model):
        thumbs_up_scores = []
        
        for keywords in topic_keywords.values():
            total_thumbs_up = 0
            for text, thumbs_up in zip(texts, thumbs_up_counts):
                words = set(text.split())
                if len(set(keywords) & words) >= 3:  # At least 3 keywords present
                    total_thumbs_up += thumbs_up
            thumbs_up_scores.append(total_thumbs_up)
        
        return self._normalize(thumbs_up_scores)

    def _calculate_sentiment(self, texts, topic_keywords, word2vec_model):
        sentiment_scores = []
        
        for keywords in topic_keywords.values():
            scores = []
            for text in texts:
                words = set(text.split())
                if len(set(keywords) & words) >= 3:
                    score = self.get_sentiment_score(text)
                    scores.append(score)
            
            avg_score = np.mean(scores) if scores else 0
            sentiment_scores.append(-avg_score)  # Invert scores to prioritize negative sentiment
        
        return self._normalize(sentiment_scores)

    def _normalize(self, values):
        if not values:
            return []
        min_val = min(values)
        max_val = max(values)
        if max_val == min_val:
            return [0.5] * len(values)
        return [(val - min_val) / (max_val - min_val) for val in values]

    def calculate_combined_scores(self, metrics):
        weights = {
            'entropy': 0.25,
            'prevalence': 0.30,
            'thumbs_up': 0.25,
            'sentiment': 0.20
        }
        
        combined_scores = []
        for i in range(len(metrics['entropy'])):
            score = sum(metrics[metric][i] * weights[metric] 
                       for metric in weights.keys())
            combined_scores.append(score)
            
        return combined_scores 