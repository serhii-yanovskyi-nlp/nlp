import os
import pickle
import re
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from fastapi import FastAPI, APIRouter
from typing import List, Any
from app.models.make_pipeline import TransformRequest
from sklearn.pipeline import make_pipeline

app = FastAPI()
model = {}

preprocesing_router = APIRouter()





class СhoiseModel:
    def __init__(self, model_name):
        if model_name in model:
            self.model = model[model_name]
        else:
            file_path = os.path.join(os.path.dirname(__file__), "models", model_name + ".model")
            with open(file_path, 'rb') as file:
                self.model = pickle.load(file)
            model[model_name] = self.model

    def transform(self, words):
        return self.model.classify_many(words)


class RemoveEmojis:
    @staticmethod
    def transform(text):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"
                                   u"\U0001F300-\U0001F5FF"
                                   u"\U0001F680-\U0001F6FF"
                                   u"\U0001F1E0-\U0001F1FF"
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)


class RemoveLinks:
    @staticmethod
    def transform(text):
        return re.sub(r'http\S+', '', text)


class Lowercase:
    @staticmethod
    def transform(text):
        return text.lower()


class Lemmatize:
    @staticmethod
    def transform(text):
        lemmatizer = WordNetLemmatizer()
        return ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(text)])


class RemoveStopwords:
    @staticmethod
    def transform(text):
        stop_words = set(stopwords.words('english'))
        return ' '.join([word for word in word_tokenize(text) if word not in stop_words])


class RemovePunctuation:
    @staticmethod
    def transform(text):
        return text.translate(str.maketrans('', '', string.punctuation))


class RemoveNumbers:
    @staticmethod
    def transform(text):
        return re.sub(r'\d+', '', text)


# Определим методы предобработки текста
preprocessing_steps = [
    RemoveEmojis,
    RemoveLinks,
    Lowercase,
    Lemmatize,
    RemoveStopwords,
    RemovePunctuation,
    RemoveNumbers
]


def create_transformer(step):
    return step.transform


@preprocesing_router.post("/transform")
async def transform(req: TransformRequest) -> Any:
    """
    Transforms the input with the provided pipeline
    """

    steps = map(create_transformer, preprocessing_steps)
    pipeline = make_pipeline(*steps)
    return pipeline.transform([req.input])[0]