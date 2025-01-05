import pandas as pd
from preprocessing import DataPreprocessor
from models import ProfileAnalyzer, SemanticMatcher, PersonalizedRecommender
from api import app
import os

def load_data():
    try:
        papers = pd.read_csv('data/raw/papers.csv')
        profiles = pd.read_csv('data/raw/user_profiles.csv')
        return papers, profiles
    except FileNotFoundError as e:
        raise RuntimeError("Required data files are missing. Ensure 'papers.csv' and 'user_profiles.csv' are in 'data/raw/'.") from e
    except pd.errors.EmptyDataError as e:
        raise RuntimeError("Data files are empty or corrupted.") from e

def preprocess_data(papers, profiles):
    try:
        preprocessor = DataPreprocessor()
        processed_papers = preprocessor.process_papers(papers)
        processed_profiles = preprocessor.process_profiles(profiles)
        return processed_papers, processed_profiles
    except Exception as e:
        raise RuntimeError("Error occurred during data preprocessing.") from e

def initialize_models():
    try:
        profile_analyzer = ProfileAnalyzer()
        semantic_matcher = SemanticMatcher()
        recommender = PersonalizedRecommender()
        return profile_analyzer, semantic_matcher, recommender
    except Exception as e:
        raise RuntimeError("Error initializing models.") from e

def save_processed_data(data, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data.to_csv(file_path, index=False)
    except Exception as e:
        raise RuntimeError(f"Error saving data to {file_path}.") from e

def main():
    try:
        papers, profiles = load_data()
        processed_papers, processed_profiles = preprocess_data(papers, profiles)
        profile_analyzer, semantic_matcher, recommender = initialize_models()

        if 'embeddings' not in processed_papers.columns:
            paper_embeddings = semantic_matcher.compute_embeddings(processed_papers['processed_text'])
            processed_papers['embeddings'] = paper_embeddings.tolist()
        
        if 'embeddings' not in processed_profiles.columns:
            profile_embeddings = semantic_matcher.compute_embeddings(processed_profiles['processed_interests'])
            processed_profiles['embeddings'] = profile_embeddings.tolist()

        save_processed_data(processed_papers, 'data/processed/processed_papers.csv')
        save_processed_data(processed_profiles, 'data/processed/processed_profiles.csv')

        print("Data preprocessing and model initialization complete.")
        app.run(debug=True)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
