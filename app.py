from flask import Flask, request
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from flask_socketio import SocketIO, emit
import pickle
import os
import asyncio

app = Flask(__name__)
socketio = SocketIO(app)

@app.route("/query", methods=['POST'])
async def ChatWithAIMSGuide():
        if request.method == 'POST':
            try:
                load_dotenv()
                query = request.form['query']

                store_name = "AIMS-Users-Guide-Images-Removed"
                if os.path.exists(f"{store_name}.pkl"):
                    with open(f"{store_name}.pkl", "rb") as f:
                        VectorStore = pickle.load(f)
                else:
                    loader = UnstructuredPDFLoader("./AIMS-Users-Guide-Images-Removed.pdf")
                    data = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                    chunks = text_splitter.split_documents(data)

                    # Vector store handling
                    embeddings = OpenAIEmbeddings(openai_api_key='sk-proj-yhNQrxJsKm9iYYAPQOeqT3BlbkFJyltg9PmgSXrmiixhAJV5')
                    VectorStore = FAISS.from_texts([t.page_content for t in chunks], embedding=embeddings)
                    with open(f"{store_name}.pkl", "wb") as f:
                        pickle.dump(VectorStore, f)

                if query:
                    # Process query (if provided)
                    # Find similar documents
                    docs = VectorStore.similarity_search(query=query)

                    # Initialize LLM and chain
                    llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key='sk-proj-yhNQrxJsKm9iYYAPQOeqT3BlbkFJyltg9PmgSXrmiixhAJV5', temperature=0, max_tokens=512, streaming=True)
                    chain = load_qa_chain(llm=llm, chain_type="stuff")

          # Run chain asynchronously and send data

                # Run chain asynchronously and send data
                    for chunk in chain.run(input_documents=docs, question=query):
                        socketio.emit("openai-data", {'data': chunk})

            except Exception as e:
                print(f"Error occurred: {e}")  # Log the error

        return 'Message Sent'  # Return an empty string
