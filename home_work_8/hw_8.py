import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import string


def remove_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_links(text):
    return re.sub(r'http\S+', '', text)


def lowercase(text):
    return text.lower()


def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(text)])


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in word_tokenize(text) if word not in stop_words])


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_numbers(text):
    return re.sub(r'\d+', '', text)


def preprocess_pipeline(text, pipeline):
    preprocessed_text = text
    for func in pipeline:
        preprocessed_text = func(preprocessed_text)
    return preprocessed_text


def tfidf_vectorize(data):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(data)
    return tfidf_matrix


def bow_vectorize(data):
    count_vectorizer = CountVectorizer()
    bow_matrix = count_vectorizer.fit_transform(data)
    return bow_matrix


pipeline_params = {
    "remove_emojis": remove_emojis,
    "remove_links": remove_links,
    "lowercase": lowercase,
    "lemmatize": lemmatize,
    "remove_stopwords": remove_stopwords,
    "remove_punctuation": remove_punctuation,
    "remove_numbers": remove_numbers
}

pipeline = [pipeline_params[key] for key in pipeline_params if pipeline_params[key]]

preprocessed_data = []

data = pd.read_csv('articles.csv')
data_list = data['text'].tolist()

for text in data_list:
    preprocessed_text = preprocess_pipeline(text, pipeline)
    preprocessed_data.append(preprocessed_text)

# TF-IDF
tfidf_matrix = tfidf_vectorize(preprocessed_data)

# BoW
bow_matrix = bow_vectorize(preprocessed_data)