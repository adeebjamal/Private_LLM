from pydantic import BaseModel

class CreateConversationRequest(BaseModel):
    title: str

class LoadMessagesRequest(BaseModel):
    conversation_id: int
    start_row: int
    end_row: int

class AskQuestionRequest(BaseModel):
    conversation_id: int
    query: str
    max_tokens: int = 500
