from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document
from concurrent.futures import ThreadPoolExecutor
import os
import re
import asyncio
import chromadb
import uuid
import hashlib

class ChromaDbManager:
    def __init__(self,model_name="paraphrase-multilingual-MiniLM-L12-v2",chunk_size=500,chunk_overlap=100,db_dir="transcription"):
        self.embedding_model = SentenceTransformer(model_name_or_path=model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        self.collection_name = "transcription"
        if not os.path.exists(db_dir):
            print(f"Creating directory {db_dir}")
            os.makedirs(db_dir)
        self.chromadb_client = chromadb.PersistentClient(path=db_dir)
        self.collection = self.chromadb_client.get_or_create_collection(name=self.collection_name)
        self.executor = ThreadPoolExecutor()
    
    def generate_unique_id(self, content, metadata):
        """Generate unique ID based on content and metadata"""
        combined = f"{content}_{metadata.get('source', '')}_{metadata.get('timestamp', '')}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def load_pdf_sync(self, file_path):
        """load PDF and return text as elements list"""
        try:
            loader = PyPDFLoader(file_path=file_path)
            pages = loader.lazy_load()
            text = "\n".join([page.page_content for page in pages])
            
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            # Poprawione wyrażenie regularne:
            # - `\[(\d{2}:\d{2})\]`: Dopasowuje znacznik czasu i przechwytuje HH:MM.
            # - `\s*`: Dopasowuje zero lub więcej białych znaków (spacje, tabulatory, nowe linie) po znaczniku czasu.
            # - `(.*?):`: Leniwie dopasowuje wszystko do pierwszego napotkanego dwukropka i spacji (zakładając "Imię Nazwisko: Treść").
            # - `(.*?)`: Leniwie dopasowuje pozostałą treść.
            # - `(?=\n?\[\d{2}:\d{2}\]|$))`: Patrzy w przód, szukając nowej linii (opcjonalnie) i kolejnego znacznika czasu LUB końca ciągu.
            pattern = r"\[(\d{2}:\d{2})\]\s*(.*?)(?=\n?\[\d{2}:\d{2}\]|$)"
            matches = re.findall(pattern=pattern, string=text, flags=re.DOTALL)
            
            documents = []
            for timestamp, content_raw in matches:
                # content_raw zawiera całą treść po znaczniku czasu, włącznie z imieniem i nazwiskiem mówcy.
                # W formacie, który wcześniej generowałem, Imię Nazwisko jest częścią treści do przechwycenia.
                doc = Document(
                    page_content=content_raw.strip(),
                    metadata={"timestamp": timestamp, "source": file_name}
                )
                documents.append(doc)
            
            if not documents:
                print(f"Warning: No segments found in '{file_path}' using the defined regex pattern.")
                print(f"Full text content (first 500 chars): {text[:500]}...")
            return documents
        except Exception as e:
            print(f"Error loading and parsing PDF '{file_path}': {e}")
            return []
    
    def check_document_exist(self, doc_id):
        """Check if document with given ID exists in the collection"""
        try:
            result = self.collection.get(ids=[doc_id])
            return len(result['ids']) > 0
        except Exception as e:
            print(f"Error checking document existence for ID '{doc_id}': {e}")
            return False
    
    async def load_pdf_async(self, file_path):
        """Async load pdf and return as list"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.load_pdf_sync, file_path)
    
    def save_to_chromadb_async(self, file_path,force_update=False):
        """processing file pdf and save embeddings to chromadb"""
        try:
            doc_name = os.path.splitext(os.path.basename(file_path))[0]
            print("processing file: ",file_path)
            documents = self.load_pdf_sync(file_path=file_path)
            print("documents loaded: ",documents)
            ids = []
            docs_to_add = []
            metadatas_to_add = []
            embeddings_to_add = []
            for doc in documents:
                doc_id = self.generate_unique_id(doc.page_content, doc.metadata)
                
                if not force_update and self.check_document_exist(doc_id):
                    print(f"Document with ID '{doc_id}' already exists in the collection. Skipping.")
                    continue
                
                ids.append(doc_id)
                docs_to_add.append(doc.page_content)
                metadatas_to_add.append(doc.metadata)
                
                embedding = self.embedding_model.encode(doc.page_content)
                embeddings_to_add.append(embedding)
                
            if not ids:
                print(f"No new documents to add from '{file_path}'.")
                return
            
            if force_update:
                self.collection.upsert(
                    ids=ids, 
                    documents=docs_to_add,
                    metadatas=metadatas_to_add,
                    embeddings=embeddings_to_add
                )
                print(f"Updated {len(documents)} documents in collection '{self.collection_name}'.")
            else:
                self.collection.add(
                    ids=ids,
                    documents=docs_to_add,
                    metadatas=metadatas_to_add,
                    embeddings=embeddings_to_add
                )
                print(f"Added {len(documents)} documents to collection '{self.collection_name}'.")
                
            collection_count = self.collection.count()
            print(f"Collection '{self.collection_name}' now contains {collection_count} documents.")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return

    def search_async(self, file_name, query, k=3):
        """Async search with optional filtering by file_name"""
        try:
            print("file name from search_async: ", file_name)
            print("query from search_async: ", query)

            query_embedding = self.embedding_model.encode(query)

            
            search_kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": k
            }

            if file_name:  
                search_kwargs["where"] = {"source": file_name}

            results = self.collection.query(**search_kwargs)

            print("Result from function search_async: ", results)

            documents = [Document(page_content=doc, metadata={"source": file_name if file_name else "unknown"}) 
                         for doc in results["documents"][0]]

            return documents

        except Exception as e:
            print(f"Error searching: {e}")
            return []
    
    # def search(self,file_name,query:str,k:int = 2):
    #     """search sync from async function search"""
    #     loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(loop=loop)
    #     return loop.run_until_complete(self.search_async(file_name=file_name,query=query,k=k))

def format_docs(docs):
    """Format documents for display"""
    return "\n\n".join(doc.page_content for doc in docs)

async def main():
    chroma_manager = ChromaDbManager()
    #print(chroma_manager.collection.peek())
    #await chroma_manager.save_to_chromadb_async("2025-04-01_14-00.pdf")
    
    # template="""
    #      Jesteś pomocnym aystentem. 
    #      Twoim zadaniem jest odpowiedzieć na
    #      pytanie na podstawie podanych informacji.
    #      <context>
    #      {context}
    #      </context>
       
    #      Odpowiedz na pytanie:
    #      {question}
    # """
    # rag_prompt = ChatPromptTemplate.from_template(template)
    # model = ChatOllama(
    #     model="mistral"
    # )
    #question="Jaki jest cel zebrania?"
    #docs = await chroma_manager.search_async(query=question,file_name="2025-04-01_14-00")
    #print(docs)
    # chain = (
    #     RunnablePassthrough.assign(context=lambda x: "\n\n".join(x["context"]))
    #     | rag_prompt
    #     | model
    #     | StrOutputParser()
    # )
    # print(chain.invoke({"context": docs, "question": question}))
    
#asyncio.run(main())