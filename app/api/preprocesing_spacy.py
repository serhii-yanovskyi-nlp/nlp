import os
import re
import string
import spacy
from fastapi import FastAPI, APIRouter
from typing import Any

from spacy import Language
from spacy.tokens import Doc

from app.models.make_pipeline import TransformRequest, Spacy
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



transpacy_router = APIRouter()


@Language.factory('replace')
class Replace(object):
    # nlp: Language

    def __init__(self, nlp: Language, name: str, old: str, new: str):
        self.nlp = nlp
        self.name = name
        self.old = old
        self.new = new

    def __call__(self, doc: Doc) -> Doc:
        text = doc.text
        return self.nlp.make_doc(text.replace(self.old, self.new))





path_model = os.path.join(os.path.dirname(__file__), "model/spacy/en_core_web_sm-3.7.1/en_core_web_sm/en_core_web_sm"
                                                     "-3.7.1")


@transpacy_router.post("/transpacy")
async def transform(req: Spacy) -> Any:
    d = req.disable
    if d is None:
        d = []

    nlp = spacy.load(path_model, disable=d)
    # print(req)
    for component in req.components:
        p = component.params
        if p is None:
            p = {}

        if component.after is None:
            nlp.add_pipe(component.factory, first=True, config=p)
        else:
            nlp.add_pipe(component.factory, after=component.after, config=p)
    doc = nlp(req.input)

    return str(doc)
