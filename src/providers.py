import os
from enum import Enum
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class EmbeddingProvider(str, Enum):
    GOOGLE = "google"
    OPENAI = "openai"


class ChatProvider(str, Enum):
    GOOGLE = "google"
    OPENAI = "openai"


def _normalize(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = value.strip().lower()
    return normalized or None


def get_embeddings(provider: Optional[str] = None):
    """
    Retorna o objeto de embeddings compatível com o provider solicitado.

    A resolução considera, nesta ordem:
      1. Parâmetro explícito.
      2. Variável de ambiente EMBEDDING_PROVIDER.
      3. Valor padrão ("google").
    """

    resolved = _normalize(provider) or _normalize(os.getenv("EMBEDDING_PROVIDER")) or EmbeddingProvider.GOOGLE.value

    if resolved == EmbeddingProvider.GOOGLE.value:
        model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
        return GoogleGenerativeAIEmbeddings(model=model)

    if resolved == EmbeddingProvider.OPENAI.value:
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model)

    raise ValueError(
        f"Embedding provider '{resolved}' não suportado. "
        "Use 'google' ou 'openai' (ajuste EMBEDDING_PROVIDER)."
    )


def get_chat_model(provider: Optional[str] = None):
    """
    Retorna o modelo de chat configurado conforme provider escolhido.

    A resolução considera, nesta ordem:
      1. Parâmetro explícito.
      2. Variável de ambiente CHAT_PROVIDER.
      3. Valor padrão ("google").
    """

    resolved = _normalize(provider) or _normalize(os.getenv("CHAT_PROVIDER")) or ChatProvider.GOOGLE.value

    if resolved == ChatProvider.GOOGLE.value:
        model = os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.5-flash-lite")
        temperature = float(os.getenv("GOOGLE_CHAT_TEMPERATURE", "0"))
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    if resolved == ChatProvider.OPENAI.value:
        model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        temperature = float(os.getenv("OPENAI_CHAT_TEMPERATURE", "0"))
        return ChatOpenAI(model=model, temperature=temperature)

    raise ValueError(
        f"Chat provider '{resolved}' não suportado. "
        "Use 'google' ou 'openai' (ajuste CHAT_PROVIDER)."
    )
