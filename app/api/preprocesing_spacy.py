import re
import string
import spacy
from fastapi import FastAPI, APIRouter
from typing import Any


from app.models.make_pipeline import TransformRequest
from sklearn.pipeline import make_pipeline

app = FastAPI()
nlp = spacy.load("en_core_web_sm")

preprocessing_router_specy = APIRouter()

class SpaCyPreprocessor:
    @staticmethod
    def remove_links(text):
        return re.sub(r'http\S+', '', text)

    @staticmethod
    def lowercase(text):
        return text.lower()

    @staticmethod
    def lemmatize(text):
        doc = nlp(text)
        tokens = [token.lemma_.lower().strip() for token in doc]
        return " ".join(tokens)

    @staticmethod
    def remove_stopwords(text):
        doc = nlp(text)
        tokens = [token.text for token in doc if not token.is_stop]
        return " ".join(tokens)

    @staticmethod
    def remove_punctuation(text):
        return text.translate(str.maketrans('', '', string.punctuation))

    @staticmethod
    def remove_numbers(text):
        return re.sub(r'\d+', '', text)

# Определим методы предобработки текста
preprocessing_steps = [
    SpaCyPreprocessor.remove_links,
    SpaCyPreprocessor.lowercase,
    SpaCyPreprocessor.lemmatize,
    SpaCyPreprocessor.remove_stopwords,
    SpaCyPreprocessor.remove_punctuation,
    SpaCyPreprocessor.remove_numbers
]

def create_transformer(step):
    return step

@preprocessing_router_specy.post("/transform_specy")
async def transform(req: TransformRequest) -> Any:
    """
    Transforms the input with the provided pipeline
    """

    steps = map(create_transformer, preprocessing_steps)
    pipeline = make_pipeline(*steps)
    return pipeline.transform([req.input])[0]
