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
    tags: list[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    document: str
    chunks_added: int
    total_documents: int
    total_chunks: int


class DocumentSummary(BaseModel):
    document: str
    chunks: int
    tags: list[str]


class DocumentsResponse(BaseModel):
    documents: list[DocumentSummary]
    total_chunks: int
    tags: list[str]


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    use_web: bool = True


class ResearchSource(BaseModel):
    kind: str
    title: str
    reference: str
    text: str


class ResearchResponse(BaseModel):
    question: str
    report: str
    verified: bool
    review_notes: str | None
    revision_count: int
    sources: list[ResearchSource]
    web_search_used: bool
    web_search_error: str | None
