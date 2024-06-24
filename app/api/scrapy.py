import os
import spacy
from fastapi import FastAPI, APIRouter
from typing import Any
from spacy import Language
from spacy.tokens import Doc

from app.models.scrapy_pipeline import Spacy

app = FastAPI()
nlp = spacy.load("en_core_web_sm")

spacy_router = APIRouter()
model = os.path.join(os.path.dirname(__file__), "model/spacy/en_core_web_sm-3.7.1/en_core_web_sm/en_core_web_sm"
                                                     "-3.7.1")


@Language.factory('replace')
class Replace(object):
    def __init__(self, nlp: Language, name: str, first: str, second: str):
        self.nlp = nlp
        self.name = name
        self.old = first
        self.new = second

    def __call__(self, doc: Doc) -> Doc:
        text = doc.text
        return self.nlp.make_doc(text.replace(self.old, self.new))




@spacy_router.post("/spacy")
async def transform(req: Spacy) -> Any:
    d = req.disable
    if d is None:
        d = []

    nlp = spacy.load(model, disable=d)
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
