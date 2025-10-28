# Ingestão e Busca Semântica com LangChain e PostgreSQL

Este projeto implementa um sistema de ingestão e busca semântica de documentos PDF utilizando LangChain, PostgreSQL com pgVector e modelos de linguagem da Google.

## Requisitos

- Docker e Docker Compose
- Python 3.9+
- Dependências listadas em `requirements.txt`

## Configuração

1. **Variáveis de Ambiente:**
   - Renomeie o arquivo `.env.example` para `.env`.
   - Defina `EMBEDDING_PROVIDER` e `CHAT_PROVIDER` com `google` ou `openai` conforme o stack desejado.
   - Preencha `GOOGLE_API_KEY` ou/ e `OPENAI_API_KEY` de acordo com os providers selecionados (modelo e temperatura também podem ser ajustados nas variáveis correspondentes).

2. **Instalar dependências:**
   ```bash
   pip install -r requirements.txt
   ```

## Ordem de execução

1. **Subir o banco de dados:**
   - O comando a seguir irá iniciar um container Docker com o PostgreSQL e a extensão pgVector.
   ```bash
   docker compose up -d
   ```

2. **Executar a ingestão:**
   - Este script irá ler o arquivo `document.pdf`, dividi-lo em chunks, gerar embeddings e armazená-los no banco de dados. Para trocar o provider de embeddings em tempo de execução, utilize `--embedding-provider google|openai`.
   ```bash
   python src/ingest.py
   ```
   ```bash
   python src/ingest.py --embedding-provider openai
   ```

3. **Iniciar o chat:**
   - Inicie o chat para fazer perguntas sobre o documento. É possível sobrepor os providers configurados via `.env` com os parâmetros `--embedding-provider` e `--chat-provider`.
   ```bash
   python src/chat.py
   ```
   ```bash
   python src/chat.py --embedding-provider openai --chat-provider openai
   ```

## Estrutura do Projeto

```
.
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── src/
│   ├── chat.py
│   ├── ingest.py
│   ├── providers.py
│   └── search.py
├── document.pdf
└── README.md
```
