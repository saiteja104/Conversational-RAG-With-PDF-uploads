# Conversational RAG With PDF Uploads and Chat History

This project is a **Conversational Retrieval-Augmented Generation (RAG)** system that allows users to upload PDFs, extract knowledge from them, and engage in a conversational Q&A with the uploaded content while retaining chat history for context.

---

## Features

- **PDF Uploads**: Users can upload one or more PDF files.
- **Knowledge Extraction**: Extracts content from PDFs and splits it into manageable chunks for embedding and retrieval.
- **RAG Pipeline**: Combines document retrieval with a history-aware question-answering system.
- **Chat History Management**: Retains context across user interactions.
- **Streamlit Interface**: Interactive web app for seamless interaction.
- **Customizable LLM Integration**: Uses GROQ LLM for generating responses.

---

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Create a `.env` file with the following content:
     ```env
     HF_TOKEN=<your-huggingface-token>
     ```

---

## Usage

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Enter your GROQ API key when prompted.
3. Upload PDF files and start asking questions about their content.

---

## Project Workflow

1. **PDF Processing**: Uploads are processed using `PyPDFLoader` to extract text.
2. **Text Splitting**: Extracted text is split into chunks using `RecursiveCharacterTextSplitter`.
3. **Vectorization**: Text chunks are converted to embeddings using `HuggingFaceEmbeddings`.
4. **Vector Store**: `Chroma` is used to store and retrieve document embeddings efficiently.
5. **RAG Pipeline**:
   - **History-Aware Question Reformulation**: Reformulates user questions with chat history for better context.
   - **Context Retrieval**: Retrieves relevant document sections.
   - **Answer Generation**: Generates concise answers using GROQ LLM.
6. **Chat History**: Chat history is maintained across interactions using `ChatMessageHistory`.

---

## Code Highlights

### Core Components

- **PDF Loading**: 
  ```python
  loader = PyPDFLoader(temppdf)
  docs = loader.load()
  ```
- **Embedding Creation**:
  ```python
  embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
  vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
  ```
- **History-Aware Retrieval**:
  ```python
  history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
  ```
- **RAG Chain**:
  ```python
  rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
  ```

---

## Future Improvements

- Add support for more document formats.
- Enhance scalability with larger vector stores.
- Integrate alternative LLMs for diverse use cases.

---

## Acknowledgments

- **LangChain** for building modular components.
- **Streamlit** for the interactive interface.
- **Hugging Face** and **GROQ** for providing LLM support.

---


