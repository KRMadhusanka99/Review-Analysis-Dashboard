import pandas as pd
import requests
from sentence_transformers import SentenceTransformer, util
from textstat import flesch_reading_ease
import re

class TopicInterpreter:
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def extract_relevant_sentences(self, df, topic_keywords):
        """Extract relevant sentences for each topic."""
        results = {}
        
        for topic, keywords in topic_keywords.items():
            relevant_sentences = []
            
            for _, row in df.iterrows():
                review_text = str(row['content']).lower()
                thumbs_up = row['thumbsUpCount']
                
                sentences = re.split(r'(?<=[.!?]) +', review_text)
                
                for sentence in sentences:
                    keywords_found = [keyword for keyword in keywords if keyword in sentence]
                    if len(set(keywords_found)) >= 5:
                        relevant_sentences.append((sentence.strip(), thumbs_up))
            
            # Sort by thumbs up count and keep top 15
            top_sentences = sorted(relevant_sentences, key=lambda x: x[1], reverse=True)[:15]
            results[topic] = [sentence for sentence, _ in top_sentences]
        
        return results

    def interpret_topics(self, relevant_sentences, topic_keywords):
        """Interpret topics using Hugging Face API."""
        data = []
        
        for topic, sentences in relevant_sentences.items():
            if sentences:
                joined_sentences = " ".join(sentences)
                
                # Generate user concerns interpretation
                concern_prompt = (
                    f"Identify and summarize the issues, problems or frustrations which users "
                    f"experienced, and identify corresponding app features and functionalities "
                    f"in these user reviews: {joined_sentences[:15]} relevant to the keywords: "
                    f"{', '.join(topic_keywords[topic])}."
                )
                
                user_concerns = self._get_mistral_response(concern_prompt)
                
                if user_concerns:
                    # Calculate scores
                    relevance = self._calculate_relevance(topic_keywords[topic], user_concerns)
                    clarity = self._calculate_clarity(user_concerns)
                    composite_score = self._calculate_composite_score(relevance, clarity)
                    
                    data.append({
                        "Topic": topic,
                        "Topic Keywords": ", ".join(topic_keywords[topic]),
                        "User Concerns": user_concerns,
                        "Relevance Score": relevance,
                        "Clarity Score": clarity,
                        "Combined Score": composite_score
                    })
                else:
                    data.append({
                        "Topic": topic,
                        "Topic Keywords": ", ".join(topic_keywords[topic]),
                        "User Concerns": "No user concerns found.",
                        "Relevance Score": 0.0,
                        "Clarity Score": 0.0,
                        "Combined Score": 0.0
                    })
        
        return pd.DataFrame(data)

    def _get_mistral_response(self, prompt):
        """Get response from Hugging Face's Mistral AI model."""
        url = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 180,
                "temperature": 0.3
            }
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                generated_text = response.json()[0].get("generated_text", "").strip()
                return generated_text.split("\n", 1)[-1].strip() if "\n" in generated_text else generated_text
        except Exception as e:
            print(f"Error calling Mistral API: {str(e)}")
        
        return None

    def _calculate_relevance(self, keywords, generated_text):
        """Calculate relevance score using cosine similarity."""
        keywords_embedding = self.model.encode(" ".join(keywords), convert_to_tensor=True)
        text_embedding = self.model.encode(generated_text, convert_to_tensor=True)
        return float(keywords_embedding @ text_embedding.T)

    def _calculate_clarity(self, text, max_clarity=100):
        """Calculate clarity score using Flesch Reading Ease."""
        clarity = flesch_reading_ease(text)
        return min(clarity / max_clarity, 1.0)

    def _calculate_composite_score(self, relevance, clarity, relevance_weight=0.7, clarity_weight=0.3):
        """Calculate composite score combining relevance and clarity."""
        return (relevance * relevance_weight) + (clarity * clarity_weight) 