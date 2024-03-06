from typing import Optional
from pydantic import BaseModel


class RecommendationRequest(BaseModel):
    krankenkassenIk: str
    diagnoseText: Optional[str] = None
    icd10Code: Optional[str] = None
