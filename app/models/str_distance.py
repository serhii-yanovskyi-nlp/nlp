from enum import Enum
from pydantic import BaseModel, Field, StrictStr


class StringDistanceAlgorithm(Enum):
    HAMMING = "hamming"
    SMITH_WATERMAN = "smith_waterman"
    NEEDLEMAN_WUNSCH = "needleman_wunsch"
    MLIPNS = "mlipns"
    LEVENSHTEIN = "levenshtein"
    GOTOH = "gotoh"
    JARO_WINKLER = "jaro_winkler"
    STRCMP95 = "strcmp95"
    DAMERAU_LEVENSHTEIN = "damerau_levenshtein"


class Distance(BaseModel):
    method: StringDistanceAlgorithm = Field(..., title="method",
                                            description="One of the algorithms supported by textdistance",
                                            example="hamming")
    line1: StrictStr = Field(..., title="line1", description="First sequence of characters", example="book")
    line2: StrictStr = Field(..., title="line2", description="Second sequence of characters", example="cook")
    similarity: float = Field(..., title="similarity", description="Similarity between two strings", example=2)
