from typing import List, Dict, Any


class ChatHistoryManager:
    """
    Manages conversational memory and message structures for UI rendering.
    """

    def __init__(self):
        self.messages: List[Dict[str, Any]] = []

    def add_user_message(self, content: str) -> None:
        """Append a user question to chat history."""
        self.messages.append({
            "role": "user",
            "content": content
        })

    def add_assistant_message(
        self,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        retrieved_documents: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Append an assistant answer with source citations to chat history."""
        self.messages.append({
            "role": "assistant",
            "content": content,
            "sources": sources or [],
            "retrieved_documents": retrieved_documents or []
        })

    def clear(self) -> None:
        """Reset conversation history."""
        self.messages.clear()

    def get_messages(self) -> List[Dict[str, Any]]:
        """Retrieve all stored chat messages."""
        return self.messages
