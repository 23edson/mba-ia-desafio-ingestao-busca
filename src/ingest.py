import argparse
import os

# Forçar o resolvedor nativo do gRPC para evitar timeouts do c-ares em alguns ambientes
os.environ.setdefault("GRPC_DNS_RESOLVER", "native")

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from providers import EmbeddingProvider, get_embeddings

# Carrega as variáveis de ambiente do arquivo .env
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


def run_ingestion(embedding_provider: str | None = None) -> None:
    # 1. Carregamento do PDF
    pdf_path = os.getenv("PDF_PATH", "document.pdf")
    docs = PyPDFLoader(pdf_path).load()

    # 2. Divisão do texto em chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = text_splitter.split_documents(docs)
    if not split_docs:
        raise SystemExit("Nenhum documento foi carregado ou dividido.")

    # 3. Geração de embeddings
    embeddings = get_embeddings(embedding_provider)

    # 4. Armazenamento no PGVector
    connection = build_pgvector_connection_url()
    collection_name = os.getenv("PG_VECTOR_COLLECTION_NAME", "document_embeddings")
    store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )

    store.add_documents(split_docs)

    print("Ingestão concluída com sucesso!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Script de ingestão de PDF para o banco vetorial.")
    parser.add_argument(
        "--embedding-provider",
        choices=[provider.value for provider in EmbeddingProvider],
        help="Provider de embeddings a ser utilizado (sobrepõe EMBEDDING_PROVIDER).",
    )
    args = parser.parse_args()

    run_ingestion(args.embedding_provider)


if __name__ == "__main__":
    main()
