import streamlit as st
import pickle
import os
import time
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredPDFLoader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback


with st.sidebar:
    st.image("./images/aims-plus.png")
    st.text("")
    st.markdown('''
                
                This app is based on AIMS user guide and answer all your queries relevant to it.
                ''')
    

    add_vertical_space(5)
    st.write("Copyright© 2024 Assurety Consulting Inc. All Rights Reserved. ")


def stream_data(chain,docs,query):
    for chunk in chain.run(input_documents=docs, question=query):
        yield chunk
        time.sleep(0.03)


def main():
    st.header("AIMS User Guide")

    load_dotenv()
      # Load PDF document
        # Text splitting
      
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
        
        
        
    query = st.text_input("Please ask your queries here")
    button = st.button("Ask")

        #st.write(query)

    if button and query:
        # Process query (if provided)
          # Find similar documents
        docs = VectorStore.similarity_search(query=query)

          # Initialize LLM and chain
        llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key='sk-proj-yhNQrxJsKm9iYYAPQOeqT3BlbkFJyltg9PmgSXrmiixhAJV5', temperature=0, max_tokens=512, streaming=True)
        chain = load_qa_chain(llm=llm, chain_type="stuff")

          # Run chain asynchronously and send data
        st.write_stream(stream_data(chain=chain,query=query,docs=docs))
        

if __name__ == '__main__':
    main()