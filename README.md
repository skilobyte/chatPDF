# chatPDF

ChatPDF app allows you to upload a PDF and chat with it using a Retrieval-Augmented Generation (RAG) pipeline.

<p align="center">
  <img src="https://github.com/skilobyte/chatPDF/blob/main/Frontend/chatPDF-demo.png" width="600" alt="ChatPDF Demo">
</p>


### How it works:

1. **PDF Upload & Chunking**  
   When you upload a PDF, the app:
   - Parses the document using `PyPDFLoader`
   - Splits it into overlapping chunks for better semantic context
   - Stores chunks in:
     - A **local keyword database** (SQLite) for BM25 retrieval
     - A **vector store** (Chroma) using MiniLM embeddings

2. **RAG**  
   Queries use an ensemble of:
   - BM25 (keyword)
   - Dense vector similarity
   - Optional query rewriting (in *fusion* mode) using GPT to expand to 4 alternate formulations

3. **Answer Generation (LLM)**  
   Top documents are passed to an OpenAI model with a prompt restricting responses to the given context.

### Modes:
- `Simple QA` → direct retrieval and answering  
- `Better QA` → query rewriting for better coverage

### How to Run

To run the chatPDF. App locally, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/skilobyte/chatPDF.git
   ```
2. **Navigate to the Directory:**
   ```bash
   cd chatPDF
   ```
3. **Install Python Dependencies:**
    ```bash
   pip install -r requirements.txt
   ```
4. **Configure OpenAI API Key in the terminal:**
    ```bash
   export OPENAI_API_KEY="sk-..."  # Replace with your actual key
   ```
    Obtain an API key from OpenAI [here](https://platform.openai.com/account/api-keys).

5. **Run Backend:**
    ```bash
   python Backend/main.py
   ```
    This will start the Backend Flask server on port  ```8000```
6. **Run Frontend (Run in a new terminal):**
    ```bash
    cd Frontend
    python -m http.server 5500     
    ```
7. **Demo the App [here](http://localhost:5500/).**

### Improvements
1. **Token-level Optimization**
    - Adapt chunk size dynamically based on the LLM's max context length (e.g., GPT-4 vs GPT-3.5).

2. **Multimodal Support**
   - Integrate OCR to process scanned PDFs.
   - Enable file types beyond PDF (images, videos, etc.)

3. **Query Understanding & Rewriting**
   - Use a smaller local LLM to rewrite or classify queries before answering.
   - Auto-switch between summarization, QA, or citation mode depending on question type.

4. **Advanced RRF Tuning**
   - Incorporate Reciprocal Rank Fusion based on question type or retriever confidence.
