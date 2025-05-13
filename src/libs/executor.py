# executor.py
import sys
from pathlib import Path

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama 
from langchain.text_splitter import CharacterTextSplitter

def carregar_dados() -> str:
    """
    Procura recursivamente por cogni_infomes.txt em todo o projeto
    e retorna seu conteúdo como string UTF-8 (ignorando bytes inválidos).
    """
    # Este arquivo está em src/executor.py, então subimos dois níveis:
    project_root = Path(__file__).resolve().parent.parent

    encontrados = list(project_root.rglob("cogni_infomes.txt"))
    if not encontrados:
        sys.exit(f"❌ Arquivo cogni_infomes.txt não encontrado em {project_root}")

    file_path = encontrados[0]
    print("→ Arquivo encontrado em:", file_path)
    return file_path.read_text(encoding="utf-8", errors="ignore")


def inicializar_llm():
    """Configura o modelo Mistral local via Ollama."""
    return Ollama(
        model="mistral:7b-instruct-q4_K_M",
        temperature=0.3,
        num_gpu=1  # se CUDA não estiver disponível, ele ignora
    )


def criar_banco_embeddings(texto: str):
    """Processa o texto e cria/carrega o banco vetorial (Chroma)."""
    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separator="\n",
    )
    texts = splitter.split_text(texto)

    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},  # força CPU
    )

    return Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory=str(Path.cwd() / "db_cogni"),
    )


def criar_chain(llm, db):
    """Configura a cadeia RAG de RetrievalQA."""
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
    )


def responder(pergunta: str, qa_chain):
    """Gera resposta e, se necessário, direciona ao time humano."""
    termos_direcionamento = [
        "orçamento",
        "falar com atendente",
        "comercial",
        "suporte técnico",
        "demonstração",
    ]
    if any(termo in pergunta.lower() for termo in termos_direcionamento):
        return {
            "resposta": "Vou conectar você ao nosso time especializado. Por favor, aguarde...",
            "direcionar": True,
        }

    resultado = qa_chain.invoke({"query": pergunta})
    return {
        "resposta": resultado["result"],
        "fontes": [doc.metadata for doc in resultado["source_documents"]],
    }
