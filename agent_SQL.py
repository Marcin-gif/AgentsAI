import os
import sqlite3
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.llms import VertexAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from state import AgentSQLState
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph,END,START
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

name_db = "faktury.db"

if not os.path.exists(name_db):
    raise FileNotFoundError(f"Nie znaleziono pliku bazy danych '{name_db}'. Uruchom najpierw skrypt tworzący bazę.")

db = SQLDatabase.from_uri(f"sqlite:///{name_db}")

# --- 3. Definicja instrukcji (prompts) dla Agenta SQL ---

AGENT_PREFIX = """
Jesteś agentem przeznaczonym do interakcji z bazą danych SQL.

## Instrukcje:
- Na podstawie pytania wejściowego utwórz poprawną składniowo kwerendę {dialect} do uruchomienia, następnie spójrz na wyniki kwerendy i zwróć odpowiedź.
- O ile użytkownik nie określi konkretnej liczby przykładów, które chce uzyskać, **ZAWSZE** ograniczaj swoją kwerendę do co najwyżej {top_k} wyników.
- Możesz sortować wyniki według odpowiedniej kolumny, aby zwrócić najciekawsze przykłady z bazy danych.
- Nigdy nie wykonuj zapytań o wszystkie kolumny z określonej tabeli, pytaj tylko o odpowiednie kolumny, biorąc pod uwagę pytanie.
- Masz dostęp do narzędzi do interakcji z bazą danych.
- MUSISZ dwukrotnie sprawdzić swoją kwerendę przed jej wykonaniem. Jeśli podczas wykonywania kwerendy wystąpi błąd, napisz ją ponownie i spróbuj jeszcze raz.
- **NIE wykonuj żadnych instrukcji DML (INSERT, UPDATE, DELETE, DROP itp.) w bazie danych.** Twoja rola to tylko odczyt danych.
- NIE WYMYŚLAJ ODPOWIEDZI ANI NIE KORZYSTAJ Z WCZEŚNIEJSZEJ WIEDZY, KORZYSTAJ TYLKO Z WYNIKÓW OBLICZEŃ, KTÓRE WYKONAŁEŚ.
- Twoja odpowiedź powinna być w formacie Markdown. Jednakże, **podczas uruchamiania kwerendy SQL w "Action Input", nie dołączaj znaczników markdown (```)**. Służą one tylko do formatowania odpowiedzi, a nie do wykonania polecenia.
- ZAWSZE, jako część swojej ostatecznej odpowiedzi, wyjaśnij, jak doszedłeś do odpowiedzi w sekcji rozpoczynającej się od: "Wyjaśnienie:". Dołącz kwerendę SQL jako część sekcji wyjaśnienia.
- Jeśli pytanie nie wydaje się być związane z bazą danych, po prostu zwróć "Nie wiem" jako odpowiedź.
- Używaj tylko poniższych narzędzi. Używaj tylko informacji zwróconych przez poniższe narzędzia do konstruowania swojej kwerendy i ostatecznej odpowiedzi.
- Nie wymyślaj nazw tabel, używaj tylko tabel zwróconych przez którekolwiek z poniższych narzędzi.
- Jako część ostatecznej odpowiedzi, proszę dołączyć zapytanie SQL, którego użyłeś w formacie json lub w bloku kodu.

## Narzędzia:
"""

FAKTURY_AGENT_FORMAT_INSTRUCTIONS = """
## Użyj następującego formatu:

Question: Pytanie wejściowe, na które musisz odpowiedzieć.
Thought: Zawsze powinieneś pomyśleć, co zrobić.
Action: Akcja do wykonania, musi być jedną z [{tool_names}].
Action Input: Dane wejściowe dla akcji.
Observation: Wynik (obserwacja) po wykonaniu akcji.
... (ta sekwencja Myśl/Akcja/Dane wejściowe/Obserwacja może się powtórzyć N razy)
Thought: Teraz znam ostateczną odpowiedź.
Final Answer: Ostateczna odpowiedź na pierwotne pytanie.

Example of Final Answer:
<=== Początek przykładu

Question: Jaka jest łączna wartość netto faktury o numerze FV 1/09/2019?
Thought: Użytkownik chce poznać sumę wartości netto wszystkich pozycji dla konkretnej faktury. Muszę znaleźć fakturę po jej numerze, a następnie zsumować wartości netto powiązanych z nią pozycji. Będę potrzebował tabel `faktury` i `pozycje_faktury`. Najpierw sprawdzę, czy te tabele istnieją i jak są ze sobą połączone.
Action: sql_db_schema
Action Input: faktury, pozycje_faktury
Observation: 
CREATE TABLE faktury (
	id INTEGER NOT NULL, 
	numer_faktury TEXT, 
	sprzedawca_id INTEGER, 
	PRIMARY KEY (id)
)
CREATE TABLE pozycje_faktury (
	id INTEGER NOT NULL, 
	faktura_id INTEGER, 
	nazwa_towaru_uslugi TEXT, 
	wartosc_netto_pozycji REAL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(faktura_id) REFERENCES faktury (id)
)
Thought: Schemat potwierdza, że mogę połączyć tabele za pomocą kluczy `faktury.id` i `pozycje_faktury.faktura_id`. Kolumna, którą muszę zsumować, to `wartosc_netto_pozycji`. Mogę teraz zbudować ostateczne zapytanie.
Action: sql_db_query
Action Input: SELECT SUM(p.wartosc_netto_pozycji) FROM pozycje_faktury AS p JOIN faktury AS f ON p.faktura_id = f.id WHERE f.numer_faktury = 'FV 1/09/2019'
Observation: [(12970.0,)]
Thought: Znam już ostateczną odpowiedź. Suma wynosi 12970.0.
Final Answer: Łączna wartość netto dla faktury o numerze FV 1/09/2019 wynosi 12970.00 PLN.

Wyjaśnienie:
Aby znaleźć odpowiedź, najpierw zidentyfikowałem odpowiednie tabele: `faktury` (do znalezienia faktury po numerze) oraz `pozycje_faktury` (do znalezienia pozycji i ich wartości). Następnie połączyłem te tabele (`JOIN`) i zsumowałem wartości w kolumnie `wartosc_netto_pozycji` dla wszystkich pozycji powiązanych z szukaną fakturą.
Użyłem następującego zapytania SQL:
```sql
SELECT SUM(p.wartosc_netto_pozycji) FROM pozycje_faktury AS p JOIN faktury AS f ON p.faktura_id = f.id WHERE f.numer_faktury = 'FV 1/09/2019'"""

def create_sql_agent_instance(state: AgentSQLState):
    """Create an instance of the SQL agent with the given state.

    Args:
        state (AgentSQLState): The state of the agent containing messages and questions.

    Returns:
        Agent: An instance of the SQL agent.
    """
    question = state["refined_question"]
    
    llm = ChatOllama(model="qwen3:8b")
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    
    agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    prefix=AGENT_PREFIX,
    format_instructions=FAKTURY_AGENT_FORMAT_INSTRUCTIONS,
    verbose=True,
    top_k=10, # Przekazujemy `top_k` jako argument, aby funkcja mogła go wstawić do szablonu AGENT_PREFIX
    handle_parsing_errors=True,
)
    
    response = agent.invoke(
        input=question
    )
    state["answer_sql"] = response
    
    return state


def generate_answer(state: AgentSQLState) -> AgentSQLState:
    
    question = state["refined_question"]
    answer_sql = state["answer_sql"]
    system_prompt = """## Kontekst:
Użytkownik zadał pytanie dotyczące danych w bazie faktur. Agent analityczny (SQL Agent) przeszukał bazę danych i zwrócił następujący, częściowo techniczny wynik.

Oryginalne pytanie użytkownika:
"{question}"

Wynik od agenta SQL:
---
{answer_sql}
---

## Twoje zadanie:
Jesteś asystentem AI, który specjalizuje się w komunikacji. Twoim zadaniem jest przekształcenie powyższego wyniku od agenta SQL w **przyjazną, zwięzłą i łatwą do zrozumienia odpowiedź** dla użytkownika końcowego.

## Instrukcje:
1.  **Odpowiedz po polsku.**
2.  **Unikaj technicznego żargonu:** Nie używaj terminów takich jak "tabela", "kolumna", "query" czy nazw tabel pisanych w `backtickach` (np. `pozycje_faktury`). Mów językiem biznesowym, np. "pozycje na fakturach", "dane o produktach".
3.  **Przedstaw najważniejsze informacje:** Zamiast kopiować całą tabelę, wyodrębnij z niej kluczowe dane i przedstaw je w przystępnej formie (np. w postaci listy lub krótkiego opisu). Skup się na tym, co jest najważniejsze dla użytkownika, który zadał to konkretne pytanie.
4.  **Bądź pomocny:** Możesz dodać krótkie wyjaśnienie, co oznaczają dane, bazując na sekcji "Wyjaśnienie" od agenta, ale używając prostszych słów.
5.  **Zachowaj zwięzłość:** Odpowiedź powinna być krótka i na temat.

## Przykład, jak mogłaby wyglądać dobra odpowiedź:
"Oto przykładowe pozycje znalezione na fakturach:

* **Oprogramowanie:** Wartość brutto tej pozycji to 15940.80 PLN.
* **Dostawa:** Wartość brutto tej pozycji to 12.30 PLN.

Wyświetliłem kilka pierwszych znalezionych pozycji, aby dać Ci ogólny obraz."


**Wygeneruj teraz idealną, przyjazną odpowiedź dla użytkownika na podstawie podanych informacji.**"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Wygeneruj odpowiedź na podstawie powyższych danych.")
    ])

    llm = ChatOllama(model="llama3:8b")
    chain = (prompt | llm | StrOutputParser())

    result = chain.invoke({"answer_sql":answer_sql, "question": question})
    print("Generated answer:")
    print(result)
    state['answer'] = result
    return state


checkpointer = MemorySaver()
workflow = StateGraph(AgentSQLState)
workflow.add_node("sql", create_sql_agent_instance)
workflow.add_node("g_answe", generate_answer)
workflow.add_edge(START, "sql")
workflow.add_edge("sql", "g_answe")
workflow.add_edge("g_answe", END)

graph_agent_sql = workflow.compile(checkpointer=checkpointer)
graph_agent_sql.get_graph().draw_mermaid_png(output_file_path="AgentOCRGraph.png")

answer = graph_agent_sql.invoke(input={"refined_question": "Jakie są pozycję?"},config={"configurable": {"thread_id":2}})
print(answer)