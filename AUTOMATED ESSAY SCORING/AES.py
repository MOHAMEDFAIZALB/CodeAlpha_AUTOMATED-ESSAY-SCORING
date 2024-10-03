# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from textstat import flesch_reading_ease, flesch_kincaid_grade, gunning_fog
from nltk import word_tokenize, sent_tokenize

nltk.download('punkt')
nltk.download('stopwords')

# Load dataset (ensure you have the dataset CSV from Kaggle)
data = pd.read_csv('essays.csv')

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.lower()  # Convert to lowercase
    return text

# Apply preprocessing
data['essay'] = data['essay'].apply(preprocess_text)

# Feature Engineering - Textual features
def extract_features(text):
    features = {}
    
    # Basic Features
    word_count = len(word_tokenize(text))
    sent_count = len(sent_tokenize(text))
    
    # Readability Scores
    fre = flesch_reading_ease(text)
    fk_grade = flesch_kincaid_grade(text)
    gf_index = gunning_fog(text)
    
    features['word_count'] = word_count
    features['sent_count'] = sent_count
    features['flesch_reading_ease'] = fre
    features['flesch_kincaid_grade'] = fk_grade
    features['gunning_fog_index'] = gf_index
    
    return pd.Series(features)

# Apply feature extraction
features_df = data['essay'].apply(extract_features)

# TF-IDF Vectorizer for n-grams
tfidf = TfidfVectorizer(max_features=500, stop_words=stopwords.words('english'))
tfidf_matrix = tfidf.fit_transform(data['essay'])

# Combine all features into a single DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
combined_features = pd.concat([features_df, tfidf_df], axis=1)

# Target (essay scores)
target = data['score']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(combined_features, target, test_size=0.2, random_state=42)

# Model: Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Example Prediction for a new essay
new_essay = "This is a sample essay to evaluate."
new_essay_preprocessed = preprocess_text(new_essay)
new_features = extract_features(new_essay_preprocessed)
new_tfidf = tfidf.transform([new_essay_preprocessed])
new_tfidf_df = pd.DataFrame(new_tfidf.toarray(), columns=tfidf.get_feature_names_out())
combined_new_features = pd.concat([new_features, new_tfidf_df], axis=1)

# Predict the score
predicted_score = model.predict(combined_new_features)
print(f'Predicted Score: {predicted_score[0]}')
