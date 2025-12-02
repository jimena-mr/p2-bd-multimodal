from pydantic import BaseModel

class SearchResponse(BaseModel):
    doc_id: str
    score: float

class SQLQuery(BaseModel):
    query: str

