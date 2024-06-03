import string
import re
from gensim.models.doc2vec import Doc2Vec
from fastapi import APIRouter
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
import os
from typing import List
import numpy as np
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def preproc(text):
    text = text.lower()
    text = ' '.join(filter(lambda word: word not in stop_words, text.split()))
    text = ''.join(char for char in text if char not in string.punctuation)
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())

    return text


def label(vector):
    similarities = [np.dot(label_vector, vector) for label_vector in label_vectors]
    closest_index = np.argmax(similarities)
    return tags[closest_index]


doc2vec_router = APIRouter()
model = os.path.join(os.path.dirname(__file__), "model", "doc2vec")
doc2vec_model = Doc2Vec.load(model)
tags = ['sport', 'tech', 'business', 'entertainment', 'politics']
label_vectors = [doc2vec_model.dv[label] for label in tags]


@doc2vec_router.post("/doc2vec")
async def classify(sentences: List[str]):
    class_sentences = []
    for sentence in sentences:
        cleaned_sentence = preproc(sentence)
        tokenized_sentence = simple_preprocess(cleaned_sentence, deacc=True)
        inferred_vector = doc2vec_model.infer_vector(tokenized_sentence)
        label = label(inferred_vector)
        class_sentences.append([sentence, label])
    return class_sentences
