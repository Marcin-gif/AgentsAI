from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import faiss

import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os
import asyncio



class FaissDbManager:
    def __init__(self,model_name="paraphrase-multilingual-MiniLM-L12-v2",chunk_size=500,chunk_overlap=100,db_dir="faiss_db_transcript"):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        self.db_dir = db_dir
        os.makedirs(db_dir,exist_ok=True)
        self.executor = ThreadPoolExecutor()
    
    def load_pdf_sync(self, file_path):
        """load PDF and return text as elemets list"""
        loader = PyPDFLoader(file_path=file_path)
        pages = loader.lazy_load()
        text="\n".join([page.page_content for page in pages])
        return self.text_splitter.split_text(text=text)
    
    async def load_pdf_async(self, file_path):
        """Async load pdf and return as list"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.load_pdf_sync, file_path)
    
    async def save_to_faiss_async(self, file_path):
        """processing file pdf and save embeddings to FAISS (each file have seperate vector db)"""
        
        doc_name = os.path.splitext(os.path.basename(file_path))[0]
        faiss_path = os.path.join(self.db_dir, f"{doc_name}.faiss")
        
        
        if os.path.exists(faiss_path):
            print(f"Vectore databse for {doc_name} already exits!")
            return
       
        texts= await self.load_pdf_async(file_path=file_path)
        
        faiss_index = await asyncio.to_thread(FAISS.from_texts, texts, self.embeddings, allow_dangerous_deserialization=True)
        
        faiss_index.save_local(faiss_path)
        print(f"Saved {len(texts)} elements from {file_path} to {doc_name}")
        
    async def search_async(self,file_name, query, k=3):
        """Async search in specific Faiss database"""
        
        faiss_path = os.path.join(self.db_dir,f"{file_name}.faiss")
        if not os.path.exists(faiss_path):
            print(f"Database doesnt exit {file_name}")
            return 
        
        faiss_index = await asyncio.to_thread(FAISS.load_local,faiss_path,self.embeddings,allow_dangerous_deserialization=True)
        
        retriever = faiss_index.as_retriever(search_kwargs={"k":k})
        
        result =await asyncio.to_thread(retriever.invoke, query)
        
        print("\n The most relevent elements")
        for i, doc in enumerate(result):
            print(f"{i+1}. {doc.page_content}\n")
        return result
    
async def main():
    faiss_manager = FaissDbManager()
    #await faiss_manager.save_to_faiss_async("transkrypcja_spotkania.pdf")
    docs = await faiss_manager.search_async(query="Czy klient prosił o dodanie funkcji?",file_name="transkrypcja_spotkania")
    print(docs)
    
asyncio.run(main())