from langchain_community.llms import ollama
from langchain.chains import retrieval_qa
from langchain.vectorstores import Chroma


# 1. Inicialização do Modelo LLM (Mistral 7B via Ollama)

llm = ollama(
    model="mistral:7b-instruct-q4_K_M",
    temperature=0.3,  # Controla a criatividade (0 = preciso, 1 = criativo)
    num_gpu=1  # Usa a GPU (RTX 4060)
)

# 2. Configuração do Sistema RAG

qa_chain = retrieval_qa.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=Chroma.as_retriever(search_kwargs={"k": 3})  # Busca 3 trechos relevantes
)

# 3. Função Principal de Interação
def gerar_resposta(pergunta: str) -> str:
    # Regras de direcionamento
    if "demonstração" in pergunta.lower():
        return "Por favor, nos informe seu email para agendarmos uma demonstração personalizada."
    
    if "suporte técnico" in pergunta.lower():
        return "Transferindo para nosso time especializado. Aguarde ou envie um email para suporte@cogni.com"
    
    # Busca e geração contextualizada
    resposta = qa_chain.invoke({"query": pergunta})["result"]
    
    # Pós-processamento
    return resposta.strip() + "\n\n[Fonte: Base de Conhecimento COGNI]"