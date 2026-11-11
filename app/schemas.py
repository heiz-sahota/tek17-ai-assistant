"""Data shapes shared across the project."""
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """One searchable piece of a document. These fields become the index fields."""
    id: str                      # stable: source + reference + content hash, so re-uploads overwrite
    source: str                  # "tek17" or "pricing"
    reference: str               # "§ 12-15" or "03_Postal_Zone_Rules / Rule Z-2"
    title: str                   # section heading
    version: str                 # regulation or document version
    role: str = "public"         # who may see it; filtered on the server
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class ChatResponse(BaseModel):
    answer: str
    citations: list[str]
    found_in_sources: bool
