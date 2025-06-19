from state import AgentState
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from state import GradeDocument
from langchain_core.messages import HumanMessage
from ChromaDbManager import ChromaDbManager
from prompts import prompt_retrieval_grade
from date_utils import extract_date_from_question
from tools import init_agent_state

def retrieve_data(state: AgentState) -> AgentState:
    """retrieve data from the documents"""
    print("---- RETRIEVE DATA ----")
    rq = state["refined_question"]
    if state["use_filtr"] == "brak":
        state = init_agent_state(rq)
        
    if hasattr(rq, "content"):
        question = rq.content.strip()
    else:
        question = str(rq).strip()
    print("Question from retrieve data:", question)
    re_question = extract_date_from_question(question=question)
    print("Re_question ",re_question)
    
    file_name = state.get("file_name", "")
    file_name = file_name.rstrip(".pdf") if file_name else ""
    
    chroma_manager = ChromaDbManager()
    docs = chroma_manager.search_async(file_name=file_name,query=re_question)
    
    
    state["documents"] = docs
    return state

def retrieval_grader(state) -> AgentState:
    """Grade document if they are relevant to the question"""
    
    llm = ChatOllama(model="mistral", temperature=0.0)
    structured_llm_router = llm.with_structured_output(GradeDocument)
    
    relevant_docs = []
    seen_contents = set()
    for doc in state["documents"]:
        doc_content = doc.page_content.strip()
        if doc_content in seen_contents:
            continue
        human_message = HumanMessage(
            content=f"Pytanie użytkownika: {state['refined_question']}\n\nDokument: {doc.page_content}"
        )
        grade_prompt = ChatPromptTemplate.from_messages([prompt_retrieval_grade, human_message])
        grader_llm = grade_prompt | structured_llm_router
        result = grader_llm.invoke({})
        print(f"Grading document: {doc.page_content[:30]}... Result: {result.score.strip()}")
    if result.score.strip().lower() == "tak":
        print(doc)
        relevant_docs.append(doc) 
            
    state["documents"] =relevant_docs
    state["proceed_to_generate"] = len(relevant_docs) > 0
    print(f"retrieval_grader: proceed_to_generate: {state["proceed_to_generate"]}")
    return state