import textdistance
from fastapi import APIRouter
from app.models.str_distance import StringDistanceAlgorithm, Distance

calculate_router = APIRouter()


@calculate_router.get("/calculate_router/{algorithm}")
async def calculate(algorithm: StringDistanceAlgorithm, str1: str, str2: str) -> Distance:
    chosen_algorithm = algorithm.value
    distance_function = getattr(textdistance, chosen_algorithm)
    similarity_score = distance_function(str1, str2)
    return Distance(method=algorithm, line1=str1, line2=str2, similarity=similarity_score)
