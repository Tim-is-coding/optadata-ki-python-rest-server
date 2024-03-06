from pydantic import BaseModel


class AiRecommondationItem(BaseModel):
    value: str
    percentage: int
