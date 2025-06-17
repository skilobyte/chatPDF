# app.py
import os, uuid, sqlite3
from pathlib import Path
from typing import List

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import sqlite3
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_community.document_transformers import (
    EmbeddingsClusteringFilter,
    EmbeddingsRedundantFilter,
)

from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_transformers import LongContextReorder


from langchain_community.retrievers import BM25Retriever

from langchain.retrievers import EnsembleRetriever
import langchain


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline

import os

from langchain.load import dumps, loads
from typing import List, Dict

from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableParallel

from re import search


from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    PromptTemplate,
)
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.load import dumps, loads
from typing import List, Dict

from langchain.chains.question_answering import load_qa_chain
from flask_cors import CORS


KEYWORD_DB = "chunks.db"   
def create_keyword_db(db_path=KEYWORD_DB):
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
            ChunkId UNINDEXED,
            Body,
            Source UNINDEXED,
            Page UNINDEXED
        );
        """
    )
    conn.commit()
    conn.close()


def insert_chunks_to_db(chunks, db_path=KEYWORD_DB):
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    for idx, c in enumerate(chunks):
        cur.execute("INSERT INTO chunks VALUES (?,?,?,?)",
                    (f"chunk_{idx}", c.page_content,
                     c.metadata.get("source","unknown"),
                     c.metadata.get("page",-1)))
    conn.commit(); conn.close()

# ------- RRF fusion helpers ------------------------ #
def _to_plain_doc(doc: Document) -> Document:
    return Document(page_content=doc.page_content, metadata=doc.metadata.copy())

def reciprocal_rank_fusion(results: List[List[Document]], k: int = 60) -> List[Document]:
    fused: Dict[str, float] = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            key = dumps(_to_plain_doc(doc))
            fused[key] = fused.get(key, 0) + 1 / (rank + k)
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return [loads(k) for k, _ in ranked]


app = Flask(__name__, static_folder="static", static_url_path="/")
CORS(app, resources={r"/*": {"origins": "*"}})         

create_keyword_db()


vectorstore   = None      
all_docs      = []       
embeddings    = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
answer_prompt = ChatPromptTemplate.from_template(
    """Answer the question **using only** the context below.
    {context}
    Question: {question}
    Answer (cite the page numbers at the end of each part of the answer):"""
)


@app.route("/upload", methods=["POST"])
def upload_file():
    global vectorstore, all_docs
    f = request.files.get("file")
    if f is None or f.filename == "":
        return jsonify(error="no file"), 400

    tmp = f"/tmp/{uuid.uuid4().hex}_{secure_filename(f.filename)}"
    f.save(tmp)

    # 1. load & chunk
    docs     = PyPDFLoader(tmp).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    chunks   = splitter.split_documents(docs)

    # 2. keyword & vector DBs
    insert_chunks_to_db(chunks)
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="chroma")
    all_docs    = chunks

    return jsonify(status="file ingested", chunks=len(chunks))
# ---------------------- Helper builders ----------------- #

def build_compressed_retriever():
    vectordb = vectorstore.as_retriever(search_kwargs={"k": 3})
    kw = BM25Retriever.from_documents(all_docs); kw.k = 3
    ensemble = EnsembleRetriever(retrievers=[vectordb, kw], weights=[0.7, 0.3])

    compressor = DocumentCompressorPipeline(
        transformers=[
            EmbeddingsRedundantFilter(embeddings=embeddings),
            LongContextReorder(),
        ]
    )
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble)

# ----------------------- Answer helpers -------------------------------- #
def answer_simple_qa(user_query: str, llm):
    simple_retriever = build_compressed_retriever()
    docs = simple_retriever.get_relevant_documents(user_query)

    chain = (
        {"context": lambda _: docs, "question": RunnablePassthrough()}
        | answer_prompt | llm | StrOutputParser()
    )
    return chain.invoke(user_query).strip()


def answer_rag_fusion(user_query: str, llm):
    retriever = build_compressed_retriever()

    # optimise query
    sys_msg = SystemMessagePromptTemplate.from_template(
        "You are a helpful assistant that generates multiple search queries.")
    usr_msg = HumanMessagePromptTemplate.from_template(
        "Generate **four** different search queries related to: {original_query}")
    prompt = ChatPromptTemplate.from_messages([sys_msg, usr_msg])
    parse = StrOutputParser() | (lambda s: [q.strip() for q in s.split("\n") if q.strip()])
    optimized = (prompt | llm | parse).invoke({"original_query": user_query})

    # RRF fusion
    def _plain(d: Document):
        return Document(page_content=d.page_content, metadata=d.metadata.copy())

    fused: Dict[str, float] = {}
    for docs in [retriever.get_relevant_documents(q) for q in optimized]:
        for rank, doc in enumerate(docs):
            key = dumps(_plain(doc))
            fused[key] = fused.get(key, 0) + 1 / (rank + 60)
    fused_docs = [loads(k) for k, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)]

    chain = (
        {"context": lambda _: fused_docs, "question": RunnablePassthrough()}
        | answer_prompt | llm | StrOutputParser()
    )
    return chain.invoke(user_query).strip()

# ---------------------- Query route --------------------- #
@app.route("/query", methods=["POST"])
def query():
    if vectorstore is None:
        return jsonify(error="upload a file first"), 400

    body = request.get_json() or {}
    user_query = body.get("query")
    mode = body.get("mode", "simple") 
    if not user_query:
        return jsonify(error="missing query"), 400

    llm = ChatOpenAI(model_name=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0.2)

    if mode == "fusion":
        answer = answer_rag_fusion(user_query, llm)
    else:
        answer = answer_simple_qa(user_query, llm)

    return jsonify(answer=answer)

# ---------------------- run ----------------------------- #
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)), debug=True)
