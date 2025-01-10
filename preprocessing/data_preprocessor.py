import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

class DataPreprocessor:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=5000)

    def clean_text(self, text):
        if not text or not isinstance(text, str):
            return ""
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
        return ' '.join(tokens)

    def process_papers(self, papers_df):
        if not isinstance(papers_df, pd.DataFrame):
            raise ValueError("Input is not a valid DataFrame.")
        
        required_columns = ['abstract', 'title']
        missing_columns = [col for col in required_columns if col not in papers_df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in papers DataFrame: {', '.join(missing_columns)}")

        papers_df['abstract'] = papers_df['abstract'].fillna('')
        papers_df['title'] = papers_df['title'].fillna('')
        
        papers_df['processed_text'] = papers_df['abstract'].apply(self.clean_text)
        papers_df['processed_title'] = papers_df['title'].apply(self.clean_text)
        
        return papers_df

    def process_profiles(self, profiles_df):
        if not isinstance(profiles_df, pd.DataFrame):
            raise ValueError("Input is not a valid DataFrame.")
        
        required_columns = ['interests', 'skills']
        missing_columns = [col for col in required_columns if col not in profiles_df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in profiles DataFrame: {', '.join(missing_columns)}")

        profiles_df['interests'] = profiles_df['interests'].fillna('')
        profiles_df['skills'] = profiles_df['skills'].fillna('')
        
        profiles_df['processed_interests'] = profiles_df['interests'].apply(self.clean_text)
        profiles_df['processed_skills'] = profiles_df['skills'].apply(self.clean_text)
        
        return profiles_df

    def vectorize_text(self, text_series):
        if text_series.isnull().any():
            raise ValueError("Text series contains NaN values.")
        return self.vectorizer.fit_transform(text_series)
