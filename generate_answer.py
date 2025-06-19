from state import AgentState
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
template = """Odpowiedz po polsku na pytanie na podstawie kontekstu i historii konwersacji. Szczególnie weź pod uwagę ostatnie pytanie do przemyślenia:

    historia konwersacji: {history}
    
    kontekst: {context}
    
    pytanie: {question}
    
"""

prompt = ChatPromptTemplate.from_template(template=template)
llm = ChatOllama(model="mistral")
rag_chain = prompt | llm

def generate_answer(state: AgentState):
    print("GENERATING ANSWER")
    if "messages" not in state or state["messages"] is None:
        raise ValueError("State must include 'messages' before generating an answer")
    print(f"Generate answer for file: {state.get('file_name', '[brak file_name]')}")

    history = state["messages"]
    documents = state["documents"]
    refined_question = state["refined_question"]
    response = rag_chain.invoke(
        {"history": history, "context": documents, "question": refined_question}
    )
    
    generation = response.content.strip()
    
    state["messages"].append(AIMessage(content=generation))
    #print(f"messages: {state['messages']}")
    state["answer"] = generation
    return state

def cannot_answer(state: AgentState) -> AgentState:
    """Handle the case where the agent cannot answer the question."""
    print("Entering cannot answer")
    if "messages" not in state or state["messages"] is None:
        state["messages"]=[]
    state["messages"].append(
            AIMessage(
                content="Przepraszam, ale nie mogę znaleźć informacji których szukasz."
            )
        )
    print(AIMessage(content="Przepraszam, ale nie mogę znaleźć informacji których szukasz."))
    state["answer"] = AIMessage(
                content="Przepraszam, ale nie mogę znaleźć informacji których szukasz."
            )
    return state

def off_topic_response(state: dict):
    """Handle the case where the question is off-topic."""
    print("Entering off topic response") 
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    state["messages"].append(AIMessage(content="Nie mogę odpowiedzieć na twoje pytanie."))
    print(AIMessage(content="Nie mogę odpowiedzieć na twoje pytanie."))
    state["answer"] = AIMessage(
                content="Przepraszam, ale nie mogę znaleźć informacji których szukasz."
            )
    return state