from state import AgentState,AgentSupervisorState
from modified_question import rewrite_question,refine_question
from router import question_classifier,on_topic_router,proceed_to_route,check_names_files,choose_agent,agent_router
from date_utils import retrieve_date
from retrieval import retrieve_data,retrieval_grader
from generate_answer import generate_answer, off_topic_response,cannot_answer
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agent_check_files import check_files
from agent_question_answer import graph_agent_qa
from agent_check_files import graph_agent_check_files
from agent_summary import graph_agent_summary
from generate_answer import off_topic_response
def init_agent_state(question: HumanMessage) -> AgentState:
    return {
        "question": question,
        "refined_question": "",
        "refined_count": 0,
        "messages": [],
        "documents": [],
        "on_topic": "",
        "proceed_to_generate": False,
        "file_name": "",
        "files_names": [],
        "specific_agent": "",
        "answer_user": "",
        "specific_file": "",
        "file_scope": "",
        "memory": {}
    }

def map_agent_state(state: dict) -> dict:
    """Map AgentSupervisorState to Agent state."""
    
    rq = state.get("refined_question")
    if isinstance(rq, str):
        rq = HumanMessage(content=rq)
    
    agent_state = {
        "refined_question": rq,
        "messages": state.get("messages", []),
    }
    return agent_state

def process_agent_result(state: AgentSupervisorState, result: dict, agent_name: str = "") -> dict:
    """
    Wspólna funkcja do przetwarzania wyników z sub-agentów
    """
    # Pobierz odpowiedź i wyczyść ją
    answer = result.get("answer", "")
    if answer and isinstance(answer, str):
        if answer.startswith("response: '") and answer.endswith("'"):
            answer = answer[11:-1]
        elif answer.startswith("response: "):
            answer = answer[10:]
    
    # Utworz AIMessage z odpowiedzi
    ai_message = answer
    
    # Połącz messages z sub-agenta z messages z supervisor state
    combined_messages = list(state.get("messages", []))
    if "messages" in result:
        combined_messages.extend(result["messages"])
    combined_messages.append(ai_message)
    
    return {
        **state, 
        "answer": answer,
        "messages": combined_messages
    }

def agent_qa(state: AgentSupervisorState):
    agent_input = map_agent_state(state)
    print("agent_input: ", agent_input["refined_question"])
    result = graph_agent_qa.invoke(agent_input)
    print("result of q&a: ", result)
    return process_agent_result(state, result, "QA")

def agent_check_files(state: AgentSupervisorState):
    agent_input = map_agent_state(state)
    result = graph_agent_check_files.invoke(agent_input)
    return process_agent_result(state, result, "CheckFiles")

def agent_summary(state: AgentSupervisorState):
    agent_input = map_agent_state(state)
    result = graph_agent_summary.invoke(agent_input)
    return process_agent_result(state, result, "Summary")

checkpointer = MemorySaver()
workflow = StateGraph(AgentSupervisorState)
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("choose_agent", choose_agent)
workflow.add_node("agent_qa", agent_qa)
workflow.add_node("agent_check_files", agent_check_files)
workflow.add_node("agent_summary", agent_summary)
workflow.add_node("off_topic_response", off_topic_response)

workflow.add_edge("rewrite_question", "choose_agent")
workflow.add_conditional_edges(
    "choose_agent",
    agent_router,
    {
        "Agent_CheckFiles": "agent_check_files",
        "Agent_AnswerQuestion": "agent_qa",
        "Agent_Summarize": "agent_summary",
        "brak": "off_topic_response",
    },
)
workflow.add_edge("agent_qa", END)
workflow.add_edge("agent_check_files", END)
workflow.add_edge("agent_summary", END)
workflow.add_edge("off_topic_response", END)
workflow.set_entry_point("rewrite_question")
graph_agent_supervisor = workflow.compile(checkpointer=checkpointer)
graph_agent_supervisor.get_graph().draw_mermaid_png(output_file_path="Agent_Supervisor.png")

# graph_agent_supervisor.invoke({"question": HumanMessage(content="Pokaż mi wszystkie pliki ?")},config={"configurable": {"thread_id":1}})