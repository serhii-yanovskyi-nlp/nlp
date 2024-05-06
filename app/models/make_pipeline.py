from typing import List, Any, Optional
from pydantic import BaseModel, Field, StrictStr


class Step(BaseModel):
    transformer: StrictStr = Field(..., title="transformer", description="A transformer to be applied",
                                   )
    params: Optional[dict] = None


class TransformRequest(BaseModel):
    input: Any = Field(..., title="input", description="Input value")
    steps: List[Step] = Field(..., title="steps", description="List of transformers")