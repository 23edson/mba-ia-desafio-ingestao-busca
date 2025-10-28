import os
# Forçar o resolvedor nativo do gRPC para evitar timeouts do c-ares em alguns ambientes
os.environ.setdefault("GRPC_DNS_RESOLVER", "native")
import argparse
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from providers import ChatProvider, EmbeddingProvider, get_chat_model, get_embeddings
from search import concatenate_context, similarity_search

load_dotenv()

# Prompt Template (mantido estático)
prompt_template = """
CONTEXTO:
{contexto_banco_de_dados}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""
prompt = ChatPromptTemplate.from_template(prompt_template)


def build_chain(embedding_provider: str | None = None, chat_provider: str | None = None):
    embeddings = get_embeddings(embedding_provider)
    llm = get_chat_model(chat_provider)

    def build_context(inputs):
        results = similarity_search(inputs["pergunta"], embeddings)
        return concatenate_context(results)

    return (
        {"contexto_banco_de_dados": build_context, "pergunta": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def main():
    parser = argparse.ArgumentParser(description="Faça perguntas ao seu documento.")
    parser.add_argument("--question", type=str, help="A pergunta que você quer fazer.")
    parser.add_argument(
        "--embedding-provider",
        choices=[provider.value for provider in EmbeddingProvider],
        help="Provider de embeddings (sobrepõe EMBEDDING_PROVIDER).",
    )
    parser.add_argument(
        "--chat-provider",
        choices=[provider.value for provider in ChatProvider],
        help="Provider de chat (sobrepõe CHAT_PROVIDER).",
    )
    args = parser.parse_args()

    chain = build_chain(args.embedding_provider, args.chat_provider)

    if args.question:
        response = chain.invoke({"pergunta": args.question})
        print(f"RESPOSTA: {response}")
        return

    print("Bem-vindo ao chat! Faça sua pergunta ou digite 'sair' para encerrar.")
    while True:
        try:
            question = input("Faça sua pergunta: ")
            if question.lower() == "sair":
                break
            response = chain.invoke({"pergunta": question})
            print(f"RESPOSTA: {response}")
        except (EOFError, KeyboardInterrupt):
            break

if __name__ == "__main__":
    main()
