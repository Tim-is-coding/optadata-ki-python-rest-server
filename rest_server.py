import uvicorn
from fastapi import FastAPI
from typing import Optional

from logic.chat_gpt_diagnone_to_icd10_code_util import ChatGPTDiagnoneToICD10CodeUtil
from model.recommondation_request import RecommendationRequest

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    # load all data into cache and init AI models to serve even the first request at maximum speed
    ChatGPTDiagnoneToICD10CodeUtil()


@app.get("/")
def hello():
    return 'Service up and running!'


@app.get("/icd10_code/")
def diagnose_to_ICD10_code(query: Optional[str] = None):
    # Process the recommendation request here
    return ChatGPTDiagnoneToICD10CodeUtil().get_icd10_code_for_diagnose("diagnose_text")



@app.post("/recommendations/")
async def create_recommendation(request: RecommendationRequest):
    # Process the recommendation request here
    return {"krankenkassenIk": request.krankenkassenIk, "diagnoseText": request.diagnoseText,
            "icd10Code": request.icd10Code}


if __name__ == '__main__':
    uvicorn.run('rest_server:app', host='0.0.0.0', port=8000)
