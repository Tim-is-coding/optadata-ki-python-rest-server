import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from logic.chat_gpt_diagnone_to_icd10_code_util import ChatGPTDiagnoneToICD10CodeUtil
from logic.product_recommender_ai_model import ProductRemcommenderAiModel
from model.ai_recommondation import AiRecommondation
from model.recommondation_request import RecommendationRequest

app = FastAPI()

# Define a list of allowed origins for CORS
allowed_origins = [
    "http://localhost:51515",
    "https://ictorious-island-0a4be4b03.4.azurestaticapps.net"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Allows specified origins
    allow_credentials=True,  # Allows cookies to be included in requests
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, etc.)
    allow_headers=["*"],  # Allows all headers
)


@app.on_event("startup")
async def startup_event():
    # load all data into cache and init AI models to serve even the first request at maximum speed
    ChatGPTDiagnoneToICD10CodeUtil()
    ProductRemcommenderAiModel()

    # execute a dry run to ensure that the models are loaded and ready to serve
    for i in range(10):
        ProductRemcommenderAiModel().recommend_hilfsmittel(
            RecommendationRequest(krankenkassenIk="105313145", bundesLand="Hessen", icd10Code="R32"))


@app.get("/")
def hello():
    return 'Dem CIO sein REST Service is up and running!'


@app.get("/icd10_code/")
def diagnose_to_icd10_code(query: Optional[str] = None):
    return ChatGPTDiagnoneToICD10CodeUtil().get_icd10_code_for_diagnose(query)


@app.post("/jens/ai/suggestions/", response_model=List[AiRecommondation])
async def create_recommendation(recommendation_request: RecommendationRequest):
    return ProductRemcommenderAiModel().recommend_hilfsmittel(recommendation_request)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Log detailed information about the validation error
    print(f"Validation Error: {exc} for request {request}")
    # You can also include more details from `exc.errors()` and `exc.body`
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )

if __name__ == '__main__':
    uvicorn.run('rest_server:app', host='0.0.0.0', port=8000)
