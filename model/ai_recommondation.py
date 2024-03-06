from typing import List
from pydantic import BaseModel

from model.ai_recommondation_item import AiRecommondationItem


class AiRecommondation(BaseModel):
    hilfsmittelNummer: AiRecommondationItem
    prices: List[AiRecommondationItem]
