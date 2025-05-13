# main.py
import sys
from libs.executor import (
    carregar_dados,
    inicializar_llm,
    criar_banco_embeddings,
    criar_chain,
    responder,
)

def main():
    # 1) Carrega o texto de conhecimento
    texto = carregar_dados()

    # 2) Inicializa o LLM
    llm = inicializar_llm()

    # 3) Cria ou carrega o banco de embeddings
    db = criar_banco_embeddings(texto)

    # 4) Monta a chain de QA
    qa_chain = criar_chain(llm, db)

    # 5) Loop de interação
    print("Chat RAG iniciado. Digite 'sair' para encerrar.")
    while True:
        pergunta = input("Você> ").strip()
        if pergunta.lower() in ("sair", "exit", "quit"):
            print("Encerrando…")
            break

        resultado = responder(pergunta, qa_chain)
        print("\nResposta> ", resultado["resposta"])
        if resultado.get("fontes"):
            print("Fontes:")
            for fonte in resultado["fontes"]:
                print("  -", fonte)
        print()

if __name__ == "__main__":
    main()