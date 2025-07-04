import streamlit as st
import os
import re
from loader import load_pdf
from embedder import split_text, store_embeddings
from retriever import load_vectordb
from qa_chain import create_qa_chain

st.set_page_config(page_title="📄 Analisador RAG de PDFs", layout="wide")

st.title("📄 Analisador de Documentos com LLM Local")
st.markdown("Faça upload de documentos e pergunte qualquer coisa com base no conteúdo.")

# Upload de arquivos
uploaded_files = st.file_uploader("📎 Envie um ou mais arquivos PDF", type="pdf", accept_multiple_files=True)

if uploaded_files:
    text = ""
    for file in uploaded_files:
        path = os.path.join("documents", file.name)
        with open(path, "wb") as f:
            f.write(file.read())
        text += load_pdf(path) + "\n"

    # Geração de embeddings
    st.info("🔄 Processando documentos...")
    chunks = split_text(text)
    store_embeddings(chunks)
    st.success("✅ Documentos processados!")

    # Criação da chain de QA
    retriever = load_vectordb()
    qa_chain = create_qa_chain(retriever)

    # Campo de pergunta
    question = st.text_input("Você:", key="user_input")

    if question:
        with st.spinner("Pensando..."):
            raw_answer = qa_chain.run(question)

            # Extração da resposta limpa
            match = re.search(r"Resposta:\s*(.*)", raw_answer, re.DOTALL | re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
            else:
                # Caso não encontre "Resposta:", usa a última linha não vazia
                lines = [line.strip() for line in raw_answer.splitlines() if line.strip()]
                answer = lines[-1] if lines else raw_answer.strip()

        # Interface tipo chat
        st.markdown(f"**Você:** {question}")
        st.markdown(f"**Sistema:** {answer}")
