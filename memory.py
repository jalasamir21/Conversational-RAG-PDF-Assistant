"""
memory.py
---------
Owns per-session chat history. This is the only place that knows how
LangChain's message-history objects work — engines and UI never touch
BaseChatMessageHistory directly, they go through MemoryStore.
"""

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory


class MemoryStore:
    """Maps session_id -> chat history. One instance per running app."""

    def __init__(self):
        self._histories: dict[str, BaseChatMessageHistory] = {}

    def get(self, session_id: str) -> BaseChatMessageHistory:
        """Get-or-create a history for this session. Signature matches what
        RunnableWithMessageHistory expects as its `get_session_history` callable."""
        if session_id not in self._histories:
            self._histories[session_id] = InMemoryChatMessageHistory()
        return self._histories[session_id]

    def clear(self, session_id: str) -> None:
        self._histories.pop(session_id, None)

    def as_history_tuples(self, session_id: str) -> list[tuple[str, str]]:
        """Convert LangChain messages into the (role, content) tuples the
        ChatEngine.get_history interface promises."""
        history = self._histories.get(session_id)
        if not history:
            return []
        return [
            ("user", m.content) if m.type == "human" else ("assistant", m.content)
            for m in history.messages
        ]
