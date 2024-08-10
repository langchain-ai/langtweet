from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from langchain_core.documents.base import Document
from typing import TypedDict, Annotated, Sequence

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


class DocSummarizerState(TypedDict):
    url: str | None
    repo_name: str | None
    documents: Sequence[Document] | None
    formatted_content: str | None
    facts: Sequence[str] | None
    tweets: str