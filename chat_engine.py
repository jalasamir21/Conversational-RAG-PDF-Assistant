"""
chat_engine.py
---------------
Defines the stable ChatEngine interface, the Chatbot facade the UI talks
to, and the concrete engines that implement it. The UI only ever imports
Chatbot — swapping SimpleFAQEngine for RagEngine (or adding a third
engine later) never requires touching app.py.
"""

from typing import List, Protocol, Tuple


class ChatEngine(Protocol):
    def answer(self, session_id: str, question: str) -> str: ...
    def get_history(self, session_id: str) -> List[Tuple[str, str]]: ...


class Chatbot:
    """Facade the UI depends on. Knows nothing about LangChain, Groq, or Chroma."""

    def __init__(self, engine: ChatEngine):
        self.engine = engine

    def ask(self, session_id: str, question: str) -> str:
        return self.engine.answer(session_id, question)

    def history(self, session_id: str) -> List[Tuple[str, str]]:
        return self.engine.get_history(session_id)


class SimpleFAQEngine:
    """Used before any PDF has been indexed. Keeps the app usable /
    gives a clear message instead of crashing when there's no RAG chain yet."""

    def __init__(self, faq_map: dict | None = None):
        self.faq_map = faq_map or {}
        self.histories: dict[str, list[tuple[str, str]]] = {}

    def answer(self, session_id: str, question: str) -> str:
        ans = self.faq_map.get(
            question.lower(),
            "Please upload a PDF first so I can answer questions about it.",
        )
        self.histories.setdefault(session_id, []).append(("user", question))
        self.histories[session_id].append(("assistant", ans))
        return ans

    def get_history(self, session_id: str):
        return self.histories.get(session_id, [])


class RagEngine:
    """Adapts a LangChain RunnableWithMessageHistory chain to the ChatEngine interface."""

    def __init__(self, rag_chain_with_history, memory_store):
        self.chain = rag_chain_with_history
        self.memory_store = memory_store

    def answer(self, session_id: str, question: str):
        response = self.chain.invoke(
            {"question": question},
            config={"configurable": {"session_id": session_id}},
        )

        print(type(response))
        print(response)

        return str(response)

    def get_history(self, session_id: str):
        return self.memory_store.as_history_tuples(session_id)
