from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from state import AgentState,DateQuery,AgentSummaryState
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from prompts import prompt_retrieve_date,prompt_extract_date,prompt_retrieve_date_for_single_file
from dotenv import load_dotenv
import re
from tools import init_agent_state, name_files
load_dotenv()


def retrieve_date(state: AgentState):
    """Retrieve the date from the question"""
    print("---- REWRITE DATE FROM QUESTION ----")
    
    
    question = state["refined_question"]
    state = init_agent_state(question)
    print(state["file_name"])
    state["file_name"] = ""
    llm = ChatOllama(model="mistral", temperature=0.0)
    structured_llm_router = llm.with_structured_output(DateQuery)
    date_prompt = ChatPromptTemplate.from_messages(
        messages=[
            ("system", prompt_retrieve_date),
            ("human", "{input}"),
        ]
    )
    date_router = date_prompt | structured_llm_router
    result = date_router.invoke({"input": question})
    state["file_name"] = result.date
    print(state["file_name"])
    return state

def retrieve_date_for_single_file(state: AgentSummaryState) -> AgentSummaryState:
    """Retrieve the date from the question"""
    print("---- REWRITE DATE FROM QUESTION ----")
    
    question = state["refined_question"]
    files = name_files()
    prompt_r_d = prompt_retrieve_date_for_single_file(files)
    
    llm = ChatOllama(model="mistral", temperature=0.0)
    structured_llm_router = llm.with_structured_output(DateQuery)
    date_prompt = ChatPromptTemplate.from_messages(
        messages=[
            ("system", prompt_r_d),
            ("human", "{input}"),
        ]
    )
    date_router = date_prompt | structured_llm_router
    result = date_router.invoke({"input": question})
    state["file_name"] = result.date
    print(result.date)
    return state

# def extract_date_from_question(question: str):
#     """retrieve sentence from word 'spotkanie' or anything else"""
#     prompt_template = ChatPromptTemplate.from_messages([
#         ("system",prompt_extract_date),
#         ("human", "{input}")
#     ])
#     model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    
#     question_router = prompt_template | model
#     response = question_router.invoke({"input": question})
#     #print(response.content.strip())
#     return response.content.strip()

def extract_date_from_question(question: str) -> str:
    """
    Usuwa fragmenty z pytania zawierające słowo 'spotkanie', daty i godziny.
    """

    text = question

    # 1. Usuń fragmenty typu "w spotkaniu z 21 marca 2025 o godzinie 12:25"
    text = re.sub(r"\b(na|w|ze)? ?spotkaniu?.*?(?=[.?!]|$)", "", text, flags=re.IGNORECASE)

    # 2. Usuń daty w formacie 03.02.2024 lub 3.2.2024
    text = re.sub(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{4}\b", "", text)

    # 3. Usuń daty w formacie ISO: 2024-02-03
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "", text)

    # 4. Usuń daty słowne, np. "21 marca 2025"
    text = re.sub(
        r"\b\d{1,2} (stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|września|października|listopada|grudnia) \d{4}\b",
        "", text, flags=re.IGNORECASE
    )

    # 5. Usuń godziny, np. "o godzinie 14:00", "godz. 12:25"
    text = re.sub(r"\bo (godzinie|godz)?\.? ?\d{1,2}[:.]\d{2}", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bgodz\.? ?\d{1,2}[:.]\d{2}", "", text, flags=re.IGNORECASE)

    # 6. Usuń nadmiarowe spacje i oczyść pytanie
    text = re.sub(r"\s{2,}", " ", text).strip()

    return text