from state import AgentState, AgentRouter, RouteQuery,AgentSupervisorState
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from pathlib import Path
from prompts import prompt_choose_agent,prompt_question_classifier
from tools import name_files

def choose_agent(state: AgentSupervisorState) -> AgentSupervisorState:
    """Choose the agent to handle question"""
    print("---- CHOOSE AGENT ----")
    question = state["refined_question"].content
    
    history =state.get("messages", [])
    
    history_text=""
    for msg in history:
        role = "U" if msg.type == "human" else "A"
        history_text += f"{role}: {msg.content}\n"
    full_user_message = (
        "Oto historia rozmowy (od najstarszych):\n\n"
        + history_text
        + f"\nAktualne pytanie użytkownika:\n{question}"
    )
    
    
    grade_prompt = ChatPromptTemplate.from_messages([prompt_choose_agent, HumanMessage(content=full_user_message)])
    llm = ChatOllama(model="llama3:8b", temperature=0.0)
    structured_llm_router = llm.with_structured_output(AgentRouter)
    agent_router = grade_prompt | structured_llm_router
    result = agent_router.invoke({})
    state["specific_agent"] = result.agent.strip()
    print(f"Agent router: {state['specific_agent']}")
    return state

# state = AgentState()
# state["refined_question"] = HumanMessage(content="Jakie są pliki ze spotkania z 25 marca?")
# choose_agent(state)


def agent_router(state: AgentSupervisorState):
    choose_agent = state["specific_agent"]
    if choose_agent == "AgentCheckFiles":
        print("Routing to check files")
        state["specific_agent"]=""
        return "Agent_CheckFiles"
    elif choose_agent == "AgentQ&A":
        print("Routing to answer question")
        state["specific_agent"]=""
        return "Agent_AnswerQuestion"
    elif choose_agent == "AgentSummary":
        print("Routing to summarize")
        state["specific_agent"]=""
        return "Agent_Summarize"
    else:
        print("Routing to off topic response")
        return "off_topic_response"
    

def check_question_files(state: AgentState) -> AgentState:
    """Check if the question refer to a specific file, multiple files or all files."""
    if state["file"] == "a":
        print("Routing to show all files")
        return "show_all_files"
    elif state["file"] == "s":
        print("Routing to show one file")
        return "show_one_file"
    elif state["file"] == "m":
        print("Routing to show multiple files with same date")
        return "show_multiple_files_with_same_date"
    else:
        print("Invalid file type")
        return state

def question_classifier(state: AgentState) -> AgentState:
    question = state["refined_question"].content
    
    # history =state.get("messages", [])
    
    # history_text=""
    # for msg in history:
    #     role = "U" if msg.type == "human" else "A"
    #     history_text += f"{role}: {msg.content}\n"
    # full_user_message = (
    #     "Oto historia rozmowy (od najstarszych):\n\n"
    #     + history_text
    #     + f"\nAktualne pytanie użytkownika:\n{question}"
    # )
    
    llm = ChatOllama(model="llama3:8b", temperature=0.0)
    #llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    structured_llm_router = llm.with_structured_output(RouteQuery)
    # route_prompt = FewShotPromptTemplate(
    #     examples=example,
    #     example_prompt=example_prompt,
    #     suffix="Question: {input}",
    #     input_variables=["input"],
    # )
    route_prompt = ChatPromptTemplate.from_messages([
        prompt_question_classifier, 
        HumanMessage(content=question)
    ])

    question_router = route_prompt | structured_llm_router
    result = question_router.invoke({"input": question})
    state["use_filtr"] = result.datasource
    print(f"question_router: {question_router.invoke({"input": question})}")
    return state

def on_topic_router(state):
    use_filtr = state["use_filtr"]
    if use_filtr == "filtr":
        print("Routing to retrieve")
        return "retrieve_date"
    elif use_filtr == "brak":
        print(f"on_topic: {use_filtr}")
        print("Routing to off topic response")
        return "retrieve_data"

def off_topic_router(state: AgentState):
    print("Entering off topic response")
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    state["messages"].append(AIMessage(content="Nie mogę odpowiedzieć na to"))
    return state


def proceed_to_route(state: AgentState) -> AgentState:
    """Route the question to the next step"""
    if state.get("proceed_to_generate", False):
        return "generate_answer"
    elif state.get("refined_count",0) >= 2:
        state["proceed_to_generate"] = False
        return "cannot_answer"
    else:
        return "refine_question"
    
def check_names_files(state: AgentState):
    file_names = name_files()
    file_name = state["file_name"].replace(".pdf", "")
    print("file_names:", file_names)
    print("file_name:", file_name)
    if file_name in file_names:
        print("route to retrive date")
        return "retrieve_data"
    else:
        print("enterning cannot answer in check files")
        return "cannot_answer"  

        