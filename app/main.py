from fastapi import FastAPI
from app.api.api import api_router
from app.api.heartbeat import heartbeat_router
from app.api.scrapy import spacy_router
from app.core.config import settings
from app.core.event_handler import start_app_handler, stop_app_handler
from app.api.calculate import calculate_router
from app.api.preprocesing import preprocesing_router
from app.api.doc2vec import doc2vec_router
from app.api.preprocesing_spacy import preprocessing_router_specy
from app.api.q_a import qa_router

app = FastAPI(title=settings.PROJECT_NAME)

app.include_router(heartbeat_router)
app.include_router(api_router, prefix=settings.API_V1_STR, tags=["ML API"])
app.include_router(calculate_router, prefix=settings.API_V1_STR, tags=["NLP"])
app.include_router(preprocesing_router, prefix=settings.API_V1_STR, tags=["NLP"])
app.include_router(doc2vec_router, prefix=settings.API_V1_STR, tags=["NLP"])
app.include_router(preprocessing_router_specy, prefix=settings.API_V1_STR, tags=["NLP"])
app.include_router(spacy_router, prefix=settings.API_V1_STR, tags=["NLP Hillel"])
app.include_router(qa_router, prefix=settings.API_V1_STR, tags=["NLP Hillel"])

app.add_event_handler("startup", start_app_handler(app, settings.MODEL_PATH))
app.add_event_handler("shutdown", stop_app_handler(app))

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")