from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pathlib import Path
from state import AgentSummaryState, AgentCheckDateQuestion
from typing import Literal
from langgraph.types import Command
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END, START
from date_utils import retrieve_date_for_single_file
from generate_answer import off_topic_response


def prompt_agent_summary(context):
    return f"""
Oto transkrypcja spotkania zespołu projektowego. Na jej podstawie napisz zwięzłe, profesjonalne podsumowanie. Uwzględnij:

- Główne tematy poruszone na spotkaniu
- Najważniejsze decyzje i ustalenia
- Zadania do wykonania i osoby za nie odpowiedzialne (jeśli wspomniane)
- Problemy lub ryzyka, które zostały omówione
- Ewentualne pytania otwarte lub punkty do omówienia na kolejnym spotkaniu

Transkrypcja:
{context}

Podsumowanie:
"""

def load_pdf_sync(file_path):
    """Load PDF and return Document objects"""
    # Utwórz pełną ścieżkę
    full_path = Path(f"F:/python-AI/transkrypcja/{file_path}.pdf")
    
    if not full_path.exists():
        print(f"File {full_path} does not exist.")
        return []
    
    try:
        # Użyj pełnej ścieżki do loadera
        loader = PyPDFLoader(file_path=str(full_path))
        pages = loader.load()
        
        # Debug - sprawdź co zwraca
        print(f"Loaded {len(pages)} pages")
        for i, page in enumerate(pages):
            print(f"Page {i+1} content length: {len(page.page_content)}")
            print(f"Page type: {type(page)}")
        
        return pages
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return []

def check_date_question(state: AgentSummaryState) -> Command[Literal["retrieve_date","off_topic_response"]]:
    """Check if the question is about a specific date."""
    print("---- CHECK DATE QUESTION ----")
    question = state["refined_question"].content.lower()
    llm = ChatOllama(model="deepseek-r1", temperature=0.7)
    structured_llm_router = llm.with_structured_output(AgentCheckDateQuestion)
    prompt_template = ChatPromptTemplate.from_messages([
        ('system', """
         # Kroki

1.  **Analiza pytania:** Przeanalizuj dane pytanie pod kątem występowania dat.
2.  **Rozpoznawanie formatu RRRR-MM-DD:** Sprawdź, czy w pytaniu występuje ciąg znaków zgodny z formatem RRRR-MM-DD.
3.  **Rozpoznawanie formatu słownego:** Sprawdź, czy w pytaniu występuje data zapisana słownie, np. "1 kwietnia 2025".
4.  **Odpowiedź:**
    *   Jeśli w pytaniu występuje data w jednym z formatów, odpowiedz "tak".
    *   W przeciwnym razie odpowiedz "nie".

# Format odpowiedzi

Odpowiedź powinna być krótkim stwierdzeniem: "tak" lub "nie".

# Przykłady

**Przykład 1**

Pytanie: "Czy faktura została wystawiona w dniu 2023-10-27?"

Odpowiedź: tak

**Przykład 2**

Pytanie: "Jakie są warunki płatności?"

Odpowiedź: nie

**Przykład 3**

Pytanie: "Kiedy odbędzie się spotkanie w sierpniu?"

Odpowiedź: tak
         
         """),
        ('human', "{input}")
    ])
    
    check_date = prompt_template | structured_llm_router
    result = check_date.invoke({"input": question})
    if result.date == "tak":
        return Command(goto="retrieve_date")
    else:
        print("Off topic response")
        return Command(goto="off_topic_response")

def summorize_file(state: AgentSummaryState):
    """Summarize the file."""
    file = state["file_name"]
    print(file)
    docs = load_pdf_sync(file)
    if not docs:
        return state
    llm = ChatOllama(model="llama3:8b", temperature=0.7)
    prompt_template = ChatPromptTemplate.from_messages([
        ('system', """
         Oto transkrypcja spotkania zespołu projektowego. Na jej podstawie napisz zwięzłe, profesjonalne podsumowanie i w języku polskim. Uwzględnij:

- Główne tematy poruszone na spotkaniu
- Najważniejsze decyzje i ustalenia
- Zadania do wykonania i osoby za nie odpowiedzialne (jeśli wspomniane)
- Problemy lub ryzyka, które zostały omówione
- Ewentualne pytania otwarte lub punkty do omówienia na kolejnym spotkaniu

Transkrypcja:
{context}

Podsumowanie:
         
         """)
    ])
    chain = create_stuff_documents_chain(llm,prompt_template)
    
    result = chain.invoke({"context": docs})
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    state["messages"].append(AIMessage(content=result))
    print("response from summorize_file: ",result)
    state["answer"] = result
    return state
    

workflow = StateGraph(AgentSummaryState)
workflow.add_node("check_date_question", check_date_question)
workflow.add_node("off_topic_response", off_topic_response)
workflow.add_node("retrieve_date", retrieve_date_for_single_file)
workflow.add_node("summarize_file", summorize_file)
workflow.add_edge(START, "check_date_question")
workflow.add_edge("retrieve_date", "summarize_file")
workflow.add_edge("summarize_file", END)
graph_agent_summary = workflow.compile(checkpointer=True)
graph_agent_summary.get_graph().draw_mermaid_png(output_file_path="AgentSummaryGraph.png")