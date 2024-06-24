from typing import List, Any, Optional
from pydantic import BaseModel, Field, StrictStr


class Step(BaseModel):
    transformer: StrictStr = Field(..., title="transformer", description="A transformer to be applied",
                                   )
    params: Optional[dict] = None


class TransformRequest(BaseModel):
    input: Any = Field(..., title="input", description="Input value")
    steps: List[Step] = Field(..., title="steps", description="List of transformers")


class Component(BaseModel):
    factory: StrictStr = Field(..., title="factory", description="A transformer to be applied",
                               example="Transformer name")
    after: Optional[str] = None
    params: Optional[dict] = None


class Spacy(BaseModel):
    input: StrictStr = Field(..., title="input", description="Input text",
                             example="There have been many great writers in the history of English literature, but there is no doubt about which writer was the greatest.")
    components: List[Component] = Field(..., title="components", description="List of transformers")
    disable: Optional[List[str]] = None
