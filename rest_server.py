import uvicorn
from fastapi import FastAPI
from typing import Optional

from logic.chat_gpt_diagnone_to_icd10_code_util import ChatGPTDiagnoneToICD10CodeUtil
from logic.product_recommender_ai_model import ProductRemcommenderAiModel
from model.recommondation_request import RecommendationRequest

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    # load all data into cache and init AI models to serve even the first request at maximum speed
    ChatGPTDiagnoneToICD10CodeUtil()
    ProductRemcommenderAiModel()


@app.get("/")
def hello():
    return 'Dem CIO sein REST Service is up and running!'


@app.get("/icd10_code/")
def diagnose_to_icd10_code(query: Optional[str] = None):
    return ChatGPTDiagnoneToICD10CodeUtil().get_icd10_code_for_diagnose(query)


@app.post("/recommendations/")
async def create_recommendation(request: RecommendationRequest):
    print()
    # Process the recommendation request here
    return {"krankenkassenIk": request.krankenkassenIk, "diagnoseText": request.diagnoseText,
            "icd10Code": request.icd10Code}


if __name__ == '__main__':
    uvicorn.run('rest_server:app', host='0.0.0.0', port=8000)
