from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb


chroma_client = chromadb.PersistentClient(path="./transcript_db")
collection = chroma_client.get_or_create_collection(name="transcripts")
model=SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def pdf_loader(file_path,doc_name):
    """Wczytuje PDF i zwraca jego zawartość jako listę stron."""
    loader = PyPDFLoader(file_path=file_path)
    pages = loader.load()

    documents = [
        Document(page_content=page.page_content, metadata={"title": doc_name, "page":i+1})
        for i, page in enumerate(pages)
    ]
    for doc in documents:
        print(f"Strona {doc.metadata['page']}: {doc.metadata}")
    return documents

def text_splitter(text):
    """Podział tekstu na mniejsze fragmenty na przetwarzania."""
    text_s = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    splits = text_s.split_text(text)
    return splits

def transform_to_embeddings(text):
    """Zamienia listę tekstu na wektory embeddingowe."""
    return model.encode(text)


def save_to_chroma(texts, embeddings, doc_name):
    """Zapisuje embeddingi i teksty do bazy chromaDB z matadanymi"""
    existing_ids = collection.get()["ids"] if collection.count() > 0 else []

    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        doc_id = f"{doc_name}_{i}"

        # Sprawdzenie, czy ID już istnieje, jeśli tak -> pomiń
        if doc_id in existing_ids:
            print(f"Dokument {doc_id} już istnieje, pomijam...")
            continue

        collection.add(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding.tolist()],
            metadatas=[{"document_name": doc_name}]
        )
        print(f"Nowy dokument zapisany: {doc_id}")




# existing_ids = collection.get()["ids"]

# # if not existing_ids:
# #     for i, (text,embedding) in enumerate(zip(text_split, embeddings)):
# #         collection.add(
# #             ids=[str(i)],
# #             documents=[text],
# #             embeddings=[embedding.tolist()]
# #         )
# #     print("New embeddings save!")

def search_query(query,doc_name=None,k=3):
    """Przeszukuje bazę pod kątem zapytania, opcjonalnie filtrująć po nazwie dokumentu"""
    if collection.count() == 0:
        print("Baza wektorowa jest pusta")
        return

    query_embeddings = transform_to_embeddings(query)
    filters = {}
    if doc_name:
        filters["document_name"]=doc_name

    results = collection.query(
        query_embeddings=[query_embeddings.tolist()],
        n_results=k,
        where=filters if doc_name else None
        )

    if not results["documents"]:
        print("Brak wynikow dla zapytania")
        return

    print("\nThe most relevant elements: \n")
    for i, doc in enumerate(results["documents"][0]):
        print(f"{i+1}. {doc}\n")



file_path = "Podsumowanie_Wynikow_Firmy.pdf"
doc_name1="transkrypcja_spotkania"
doc_name2="Podsumowanie_Wynikow_Firmy"

# pages = pdf_loader(file_path,doc_name=doc_name)
# #print(pages)
# text_split = text_splitter("\n".join([page.page_content for page in pages]))
# #print(text_split)
# embeddings = transform_to_embeddings(text_split)
# save_to_chroma(texts=text_split,embeddings=embeddings,doc_name=doc_name)

search_query("Czy klient prosił o dodanie funkcji ?",doc_name=doc_name2,k=1)