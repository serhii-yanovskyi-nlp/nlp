import json
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = nltk.word_tokenize(text.lower())
    cleaned_text = ' '.join(lemmatizer.lemmatize(word) for word in words if word not in stop_words)
    return cleaned_text


def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        json_objects = []
        while True:
            line = file.readline()
            if not line:
                break
            try:
                json_object = json.loads(line)
                json_objects.append(json_object)
            except json.JSONDecodeError:
                continue
    return json_objects


file_path = 'combined.json'
data = read_json_file(file_path)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

num_clusters = 10
documents = [item['contents'] for item in data]
processed_docs = [preprocess_text(doc) for doc in documents]

# Bag-of-Words representation
bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(processed_docs)

# TF-IDF representation
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(processed_docs)

kmeans_bow = KMeans(n_clusters=num_clusters)
kmeans_bow.fit(bow_matrix)
bow_clusters = kmeans_bow.labels_
kmeans_tfidf = KMeans(n_clusters=num_clusters)
kmeans_tfidf.fit(tfidf_matrix)
tfidf_clusters = kmeans_tfidf.labels_

result = pd.DataFrame()
result["number_doc"] = range(len(documents))
result["Bag-of-Words"] = bow_clusters
result["TF-IDF"] = tfidf_clusters

print(result)