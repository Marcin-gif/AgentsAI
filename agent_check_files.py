"""
check_question => 
if question is a => show_all_files
if question is s => show_one_file
if question is m => show_multiple_files_with_same_date
"""

from state import AgentState, AgentRouterFiles,AgentCheckFilesState
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate
from prompts import prompt_check_files,prompt_agent_router_files
from state import AgentCheckFiles
from date_utils import retrieve_date_for_single_file
from langchain_ollama import ChatOllama
from router import name_files
from langgraph.types import interrupt,Command,Interrupt
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from agent_summary import summorize_file
import uuid

def add_message(state, msg):
    """Add message to the state."""
    if "messages" not in state:
        state["messages"] = []
    state["messages"].append(msg)
    return state

def get_valid_input(prompt, valid_options):
    """Get valid input from the user."""
    while True:
        answer = input(prompt)
        if answer in valid_options:
            return answer
        else:
            print(f"Invalid input. Please choose from {valid_options}.")


def check_files(state: AgentCheckFilesState) -> AgentCheckFilesState:
    """Generate answer to the question."""
    answer_user = state.get("answer_user")

    if answer_user:
        question = answer_user
    else:
        question = state["refined_question"].content    
    human_message = HumanMessage(
        content=f"Użytkownik: {question}"
    )

    files = name_files()
    system_prompt = prompt_check_files(available_files=files,user_question=question)
    grade_prompt = ChatPromptTemplate.from_messages([system_prompt, human_message])
    llm = ChatOllama(model="mistral",temperature=0.0)
    agent_router = grade_prompt | llm
    result = agent_router.invoke({})
    state["file_scope"] = result.content
    
    #add_message(state=state,msg=HumanMessage(content=question))
    #add_message(state=state, msg=AIMessage(content=result.content))
    
    
    #print(f"File scope: {state["file_scope"]}")
    
    return state

def generate_answer(state: AgentCheckFilesState) -> AgentCheckFilesState:
    """Generate a response for the user based on their question and available files."""
    file_scope = state.get("file_scope", "")
    question = state.get("answer") or state["refined_question"].content
    
    if not file_scope:
        response_text = "Nie znalazłem żadnych plików odpowiadających Twojemu zapytaniu."
    else:
        system_message = SystemMessage(content="Jesteś asystentem, który pomaga użytkownikowi na podstawie listy dostępnych plików.")
        human_message = HumanMessage(
            content=(
                f"Użytkownik zadał pytanie:\n'{question}'\n\n"
                f"Dostępne pliki to:\n{file_scope}\n\n"
                "Proszę wygeneruj odpowiedź dla użytkownika uwzględniając jego pytanie i dostępne pliki."
            )
        )
        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

        llm = ChatOllama(model="llama3:8b", temperature=0.0)
        chat_chain = chat_prompt | llm
        result = chat_chain.invoke({})

        response_text = result.content
    state["answer"] = response_text
    add_message(state, AIMessage(content=response_text))
    print(response_text)  # możesz usunąć lub zostawić
    
    return state
    
# def route_to_generate_answer(state: AgentCheckFilesState) -> AgentCheckFilesState:
#     """create answer to route to define if user ask one file or multiple files"""
#     answer = input("Jeśli chcesz zobaczyć konkretny plik wybierz z listy, mogę też podać z konkretnego miesiąca.")
#     state["answer_user"] = answer
#     answer_ai = state["file_scope"]
#     answer = state["answer_user"]
#     system_prompt = prompt_agent_router_files(ai_answer=answer_ai,user_question=answer)
#     human_message = HumanMessage(
#         content=f"Użytkownik: {answer}"
#     )
#     grade_prompt = ChatPromptTemplate.from_messages([system_prompt, human_message])
#     llm = ChatOllama(model="mistral", temperature=0.0)
#     structured_llm_router = llm.with_structured_output(AgentRouterFiles)
#     agent_router = grade_prompt | structured_llm_router
#     result = agent_router.invoke({})
#     state["specific_file"] = result.file.strip()
#     return state

# def route_single_multiple_files(state: AgentCheckFilesState):
#     """Route to single or multiple files."""
#     if state["specific_file"] == "s":
#         print("Routing to show one file")
#         return "single_file"
#     elif state["specific_file"] == "m":
#         print("Routing to show multiple files with same date")
#         return "check_files"
#     else:
#         print("Invalid file type")
#         return "check_files"


# def single_file(state: AgentCheckFilesState) -> Command:
#     """Ai ask user if he want to see summary or content of the file"""
#     state["file_name"] = retrieve_date_for_single_file(state)
#     print(f"File name: {state['file_name']}")
#     ai_prompt = (
#         f"Czy chciałbyś/chciałabyś zobaczyć:\n"
#         "1. Podsumowanie\n"
#         "2. Treść pliku"
#     )
    
#     # Dodaj komunikat do historii (opcjonalne)
   
#     add_message(state=state,msg=ai_prompt)
    
#     answer_user = get_valid_input(ai_prompt, ["1", "2","3", "podsumowanie", "treść pliku","pliki","plik"])
#     if answer_user in ("1", "podsumowanie","Podsumowanie"):
#         return Command(goto="summary_file", update={"file_name": state["file_name"]})
#     elif answer_user in ("2", "treść pliku", "Treść pliku"):
#         return Command(goto="content_file", update={"answer_user": "treść pliku"})
#     else:
#         return Command(goto="check_files", update={"answer_user": "check_files"})
    

# def multiple_files(state: AgentCheckFilesState) -> AgentCheckFilesState:
#     """Generate route to extract date from the question"""
#     print("Multiple files")
#     return state  
    
# def summary_file(state: AgentCheckFilesState) -> AgentCheckFilesState:
#     print("Summary file")
#     return state

# def content_file(state: AgentCheckFilesState) -> AgentCheckFilesState:
#     print("Content file")
#     return state


    
# state = AgentState()
# state["refined_question"] = HumanMessage(content="czy jest spotkanie z marca?")
# state = single_file(state)
checkpointer = MemorySaver()
builder = StateGraph(AgentCheckFilesState)

# --- Główne węzły ---
builder.add_node("check_files", check_files)
builder.add_node("generae_answer", generate_answer)

builder.add_edge(START, "check_files")
builder.add_edge("check_files", "generae_answer")
builder.add_edge("generae_answer", END)
# builder.add_node("route_to_generate_answer", route_to_generate_answer)

# builder.add_node("route_single_multiple_files", route_single_multiple_files)

# # --- Obsługa jednego pliku ---
# builder.add_node("single_file", single_file)
# builder.add_node("multiple_files", multiple_files)
# builder.add_node("summary_file", summorize_file)
# builder.add_node("content_file", content_file)

# # Możesz dodać później:
# # builder.add_node("multiple_files", multiple_files)
# # builder.add_node("show_all_files", show_all_files)

# # --- Połączenia ---
# builder.set_entry_point("check_files")

# # routing po klasyfikacji pytania
# builder.add_edge("check_files", "route_to_generate_answer")

# # obsługa jednego pliku
# builder.add_conditional_edges(
#     "route_to_generate_answer", 
#     route_single_multiple_files,
#     {
#         "single_file": "single_file",
#         "multiple_files": "multiple_files",
#         "check_files": "check_files",
#     }

# )
# builder.add_conditional_edges
# # zakończenia
# builder.add_edge("summary_file", END)
# builder.add_edge("content_file", END)
# builder.add_edge("multiple_files", END)

# --- Kompilacja ---
graph_agent_check_files = builder.compile(checkpointer=True)

"""
TODO:
 - create a function to generate a summary of the file
 - improve function to check choice of the user
 - add route to main graph
"""