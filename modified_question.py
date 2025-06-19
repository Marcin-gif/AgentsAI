from state import AgentState
from prompts import messages_rewrite_question, prompt_refine_question
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

messages_rewrite_question = [
    SystemMessage(content="""
Agent AI ma za zadanie modyfikować pytania zadane przez użytkownika w taki sposób, aby były one bardziej precyzyjne i zrozumiałe dla innego agenta AI. Celem jest poprawa efektywności przetwarzania zapytań bez zmiany ich pierwotnego znaczenia. Agent powinien skupić się na poprawie gramatyki, składni oraz na doprecyzowaniu niejasnych sformułowań.

# Kroki

1.  **Analiza Pytania Użytkownika:**
    *   Zidentyfikuj główne zapytanie i intencję użytkownika.
    *   Określ, które elementy pytania są niejasne, nieprecyzyjne lub mogą być źle zinterpretowane.
2.  **Przeformułowanie Pytania:**
    *   Popraw gramatykę i składnię, aby pytanie było poprawne językowo.
    *   Użyj bardziej precyzyjnych sformułowań, aby uniknąć dwuznaczności.
    *   Rozbij złożone pytania na prostsze, bardziej zrozumiałe fragmenty, jeśli to konieczne.
3.  **Zachowanie Znaczenia:**
    *   Upewnij się, że przeformułowane pytanie zachowuje oryginalne znaczenie i intencję użytkownika.
    *   Unikaj dodawania nowych informacji lub zmiany kontekstu pytania.

# Format Wyjściowy

Agent powinien zwrócić przeformułowane pytanie w formie pojedynczego zdania lub krótkiego akapitu.

# Przykłady

**Przykład 1**

*   Pytanie użytkownika: `czy ta rzecz ma dobry bajer?`
*   Przeformułowane pytanie: `Czy ten produkt charakteryzuje się wysoką jakością i użytecznością?`

**Przykład 2**

*   Pytanie użytkownika: `jak zrobić żeby on szybciej chodził ten komp?`
*   Przeformułowane pytanie: `Jak mogę zwiększyć szybkość działania mojego komputera?`

**Przykład 3**

*   Pytanie użytkownika: `ile kosztuje ta czerwona sukienka tam w oknie?`
*   Przeformułowane pytanie: `Jaka jest cena czerwonej sukienki wystawionej na wystawie sklepowej?`

# Uwagi

*   Agent powinien być w stanie radzić sobie z pytaniami z błędami językowymi, slangiem i potocznymi wyrażeniami.
*   Agent powinien unikać dodawania własnych opinii lub interpretacji do pytania.
*   Agent powinien skupić się na poprawie zrozumiałości pytania dla innego agenta AI, a nie na udzielaniu odpowiedzi na pytanie.
""")
]

def rewrite_question(state: dict) -> dict:
    print("---- REWRITE QUESTION ----")

    state["refined_question"] = ""
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []

    # if state["question"] not in state["messages"]:
    #     state["messages"].append(state["question"])

    if "question" not in state or state["question"] is None:
        print(type(state))
        raise ValueError("Brakuje pola 'question' w stanie agenta.")

    if len(state["messages"]) > 1:
        conversation = state["messages"][:-1]
        current_question = state["question"]

        # Zbuduj pełną historię wiadomości do modelu
        full_messages = messages_rewrite_question + conversation + [current_question]

        llm = ChatOllama(model="mistral", temperature=0.0)
        response = llm.invoke(full_messages)

        better_question = response.content.strip()
        print(f"Better question: {better_question}")
        state["refined_question"] = HumanMessage(content=better_question)
    else:
        state["refined_question"] = state["question"]
        print(state["refined_question"].content.strip())

    return state

def refine_question(state: AgentState):
    print("REFINE QUESTION AND CHECK AMOUNT OF REPHREASE")
    refined_count = state["refined_count"]
    print(f"refined_count: {refined_count}")
    if refined_count >=2:
        print("Maximum rephrease attempts reached")
        return state
    question_to_refine = state["refined_question"]
    
    human_message = HumanMessage(
        content=f"Orginalne pytanie: {question_to_refine}\n\n Dostarcz zoptymalizowane pytanie "
    )
    refine_prompt = ChatPromptTemplate.from_messages([prompt_refine_question,human_message])
    llm = ChatOllama(model="mistral")
    prompt = refine_prompt.format()
    response = llm.invoke(prompt)
    refined_question = response.content.strip()
    print(f"refined question: {refined_question}")
    state["refined_question"] = refined_question
    state["refined_count"] = refined_count+1
    return state
