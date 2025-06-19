from state import AgentState
from modified_question import rewrite_question,refine_question
from router import question_classifier,on_topic_router,proceed_to_route,check_names_files,choose_agent,agent_router
from date_utils import retrieve_date
from retrieval import retrieve_data,retrieval_grader
from generate_answer import generate_answer, off_topic_response,cannot_answer
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END,START
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
workflow = StateGraph(AgentState)

# workflow.add_node("rewrite_question",rewrite_question)
workflow.add_node("question_classifier",question_classifier)
workflow.add_node("on_topic_router",on_topic_router)
workflow.add_node("retrieve_date",retrieve_date)
workflow.add_node("retrieve_data",retrieve_data)
workflow.add_node("retrieval_grader",retrieval_grader)
workflow.add_node("generate_answer",generate_answer)
workflow.add_node("refine_question",refine_question)
workflow.add_node("cannot_answer",cannot_answer)

workflow.add_edge(START,"question_classifier")
workflow.add_conditional_edges(
    "question_classifier",
    on_topic_router,
    {
        "retrieve_date": "retrieve_date",
        "retrieve_data": "retrieve_data",
    },
)
workflow.add_conditional_edges(
    "retrieve_date",
    check_names_files,
    {
        "retrieve_data": "retrieve_data",
        "cannot_answer": "cannot_answer",
    },
)

workflow.add_edge("retrieve_data","retrieval_grader")
workflow.add_conditional_edges(
    "retrieval_grader",
    proceed_to_route,
    {
        "generate_answer": "generate_answer",
        "refine_question": "refine_question",
        "cannot_answer": "cannot_answer",
    },
)
workflow.add_edge("refine_question","retrieve_data")
workflow.add_edge("generate_answer",END)
workflow.add_edge("cannot_answer",END)
graph_agent_qa = workflow.compile(checkpointer=True)

graph_agent_qa.get_graph().draw_mermaid_png(output_file_path="RAG_Agent.png")

# input = {"question": HumanMessage(content="Jaki jest cel zebrania? w spotkaniu z 1 kwietnia 2025 o godzinie 14:00")}
# graph_agent_qa.invoke(input=input,config={"configurable": {"thread_id":1}})

# input = {"question": HumanMessage(content="Co było w spotkaniu?")}
# graph_agent_qa.invoke(input=input,config={"configurable": {"thread_id":2}})