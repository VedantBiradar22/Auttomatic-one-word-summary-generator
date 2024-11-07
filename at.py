!pip install pandas scikit-learn nltk
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download nltk data
nltk.download('punkt')
nltk.download('stopwords')

# Data preprocessing
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Load dataset (replace with actual path to your dataset)
# Load dataset 
df = pd.read_csv('D:\vedant/stackoverflow_sample.csv') # Replace with the actual path

# Apply preprocessing
df['processed_text'] = df['title'] + " " + df['body']
df['processed_text'] = df['processed_text'].apply(preprocess_text)

# Convert tags to a single string
df['tags'] = df['tags'].apply(lambda x: ' '.join(eval(x)))

# Split the data
X = df['processed_text']
y = df['tags']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# k-Nearest Neighbors model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_vec, y_train)
y_pred_knn = knn.predict(X_test_vec)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print(f'k-NN Accuracy: {knn_accuracy * 100:.2f}%')

# Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_vec, y_train)
y_pred_rf = rf.predict(X_test_vec)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Accuracy: {rf_accuracy * 100:.2f}%')

# Function to predict tags for new questions
def predict_tags(question, model):
    processed_question = preprocess_text(question)
    vec_question = vectorizer.transform([processed_question])
    predicted_tags = model.predict(vec_question)
    return predicted_tags[0]

# Example usage
new_question = "How to implement a binary search tree in Python?"
predicted_tags_knn = predict_tags(new_question, knn)
predicted_tags_rf = predict_tags(new_question, rf)
print(f'Predicted tags by k-NN: {predicted_tags_knn}')
print(f'Predicted tags by Random Forest: {predicted_tags_rf}')
