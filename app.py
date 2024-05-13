import streamlit as st
import pickle
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback


with st.sidebar:
    st.title("ChatPDF")
    st.markdown('''
                This app is an LLM powered chatbot
                ''')
    

    add_vertical_space(5)
    st.write("Made by Ali Asad")


def main():
    st.header("Chat with PDF")

    load_dotenv()

    pdf = st.file_uploader("Upload your PDF",type ="pdf")
    
    if pdf is not None:
        
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text+=page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )
        chunks = text_splitter.split_text(text=text)
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                VectorStore = pickle.load(f)
            st.write("Embeddings Loaded from the Disk")

        else:
            embeddings = OpenAIEmbeddings(openai_api_key='sk-proj-yhNQrxJsKm9iYYAPQOeqT3BlbkFJyltg9PmgSXrmiixhAJV5')
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(VectorStore,f)


        query = st.text_input("Ask questions about your PDF")
        button = st.button("Ask")

        #st.write(query)

        if button and query:
            docs = VectorStore.similarity_search(query=query) 
            
            llm = OpenAI(model_name="gpt-3.5-turbo",openai_api_key='sk-proj-yhNQrxJsKm9iYYAPQOeqT3BlbkFJyltg9PmgSXrmiixhAJV5')
            chain = load_qa_chain(llm=llm,chain_type="stuff")
            
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs,question=query)
                print(cb)
            
            st.write(response)
            


        #st.write(chunks)    
        

if __name__ == '__main__':
    main()