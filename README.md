# Analisador de Documentos com IA (RAG Local) 🧠📄

Este é um projeto desenvolvido para **responder automaticamente a perguntas com base em PDFs carregados pelo usuário**, utilizando a arquitetura RAG (Retrieval-Augmented Generation).  
Tudo é feito com **modelos open-source rodando localmente**, sem custos com API.

---

## 📋 Descrição do Projeto

A aplicação permite:
- Fazer upload de um ou mais arquivos PDF.
- Processar os documentos e gerar embeddings semanticamente relevantes.
- Fazer perguntas em linguagem natural sobre o conteúdo dos PDFs.
- Obter respostas geradas por uma IA local baseada em LLM.
- Visualizar as perguntas e respostas com uma interface em formato de chat (Você / Sistema).

A arquitetura RAG combina **busca vetorial com geração por linguagem natural**, oferecendo respostas contextualizadas com base no conteúdo dos documentos enviados.

---

## 🚀 Funcionalidades

1. **Upload de PDFs**  
   O usuário pode carregar documentos para análise. Os textos são automaticamente extraídos e divididos em partes para indexação.

2. **Busca semântica com Embeddings Locais**  
   O sistema gera vetores a partir dos textos dos PDFs e armazena-os em uma base vetorial local (ChromaDB), sem uso de nuvem.

3. **Geração de Respostas com LLM Local**  
   As respostas são geradas usando o modelo open-source `TinyLlama-1.1B-Chat-v1.0`, diretamente na CPU da máquina do usuário.

4. **Interface em Estilo de Chat**  
   O usuário interage com o sistema em uma interface simples e elegante, semelhante a um chat, feita com Streamlit.

---

## 🛠️ Tecnologias Utilizadas

- **Python**
- **Streamlit**: Interface gráfica web local
- **LangChain**: Orquestração da arquitetura RAG
- **Transformers (HuggingFace)**: Modelo LLM local (TinyLlama)
- **ChromaDB**: Banco vetorial local
- **PyMuPDF**: Leitura e extração de texto de PDFs
- **sentence-transformers**: Embeddings locais

---
