from pydantic import BaseModel, Field


class Source(BaseModel):
    document: str
    chunk_index: int
    score: float
    text: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=10)


class QueryResponse(BaseModel):
    question: str
    answer: str | None
    grounded: bool
    note: str | None
    sources: list[Source]


class IngestTextRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)


class IngestResponse(BaseModel):
    document: str
    chunks_added: int
    total_documents: int
    total_chunks: int


class DocumentSummary(BaseModel):
    document: str
    chunks: int


class DocumentsResponse(BaseModel):
    documents: list[DocumentSummary]
    total_chunks: int
