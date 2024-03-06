from typing import Optional
from pydantic import BaseModel


class RecommendationRequest(BaseModel):
    krankenkassenIk: str
    bundesLand: str
    icd10Code: Optional[str] = None
