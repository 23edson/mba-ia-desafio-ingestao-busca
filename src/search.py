import os
from typing import Iterable, Tuple

from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_core.documents import Document

load_dotenv()


def build_pgvector_connection_url() -> str:
    connection_url = os.getenv("PGVECTOR_URL")
    if connection_url:
        return connection_url

    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    user = os.getenv("PGUSER", "postgres")
    password = os.getenv("PGPASSWORD", "postgres")
    database = os.getenv("PGDATABASE", "rag")

    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"


def similarity_search(query, embeddings, k=10):
    """
    Realiza uma busca por similaridade no banco de dados vetorial.

    Args:
        query (str): A pergunta do usuário.
        embeddings: O objeto de embeddings.
        k (int): O número de resultados a serem retornados.

    Returns:
        list: Uma lista de documentos similares.
    """
    connection = build_pgvector_connection_url()
    collection_name = os.getenv("PG_VECTOR_COLLECTION_NAME", "document_embeddings")

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )

    results = vectorstore.similarity_search_with_score(query, k=k)
    """ for i, (doc, score) in enumerate(results, start=1):
        print("="*50)
        print(f"Resultado {i} (score: {score:.2f}):")
        print("="*50)

        print("\nTexto:\n")
        print(doc.page_content.strip())

        print("\nMetadados:\n")
        for k, v in doc.metadata.items():
            print(f"{k}: {v}")
 """
    return results


def concatenate_context(results: Iterable[Tuple[Document, float]]) -> str:
    """
    Concatena o conteúdo dos documentos retornados pela busca para compor o contexto.
    """
    contents = [
        doc.page_content.strip()
        for doc, _ in results
        if getattr(doc, "page_content", "").strip()
    ]
    return "\n\n".join(contents)
