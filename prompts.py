from langchain_core.prompts import ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage

example_prompt = PromptTemplate.from_template("Question: {question}\n\nAnswer: {answer}")
example = [
    {
        "question": "Czy klient prosił o dodanie funkcji? czy były błędy krytyczne? Ze spotkania z 2025-03-21 o godzinie 12:25",
        "answer": "tak",
    },
    {
        "question": "Co jest stolicą Polski?",
        "answer": "nie",
    },
    {
        "question": "Czy spotkanie odbyło się 2025-03-21 o godzinie 12:25?",
        "answer": "tak",
    },
    {
        "question": "Jakie są najnowsze zmiany w projekcie?",
        "answer": "nie",
    },
    {
        "question": "Czy mogę prosić o podsumowanie spotkania z 2025-03-21?",
        "answer": "tak",
    },
    {
        "question": "Jakie są cele projektu?",
        "answer": "nie",
    },
    {
        "question": "Czy były jakieś problemy z klientem? w spotkaniu z 21 marca 2025 o godzinie 13:00",
        "answer": "tak",
    },
    {
        "question": "Co było na spotkaniu?",
        "answer": "nie",
    },
    {
        "question": "Jaka była pogoda w polsce 21 marca 2025?",
        "answer": "nie",
    },
    {
        "question": "Jaka była pogoda w polsce 01-02-2024?",
        "answer": "nie",
    },
    {
        "question": "Co to jest kot?",
        "answer": "nie",
    },
    {
        "question": "Co było w spotkaniu?",
        "answer": "nie",
    },
]

messages_rewrite_question = [
            SystemMessage(
                content="Jesteś ekspertem, kóry przekształca pytania użytkownika, aby uzyskać lepsze pytania do zoptymalizowania wyszukiwania."
            )
        ]

prompt_choose_agent = SystemMessage(content="""
Jesteś agentem koordynującym, który analizuje pytanie użytkownika dotyczące spotkań i plików PDF. Twoim zadaniem jest rozpoznać intencję pytania i przekierować je do odpowiedniego agenta pomocniczego.

Dostępni agenci:

1. **AgentCheckFiles** – zwraca liczbę i listę plików PDF dotyczących spotkań.
2. **AgentQ&A** – odpowiada na pytania użytkownika na podstawie treści konkretnego pliku PDF.
3. **AgentSummary** – generuje zwięzłe podsumowanie ze wskazanego pliku PDF.
4. **brak** – jeśli pytanie nie dotyczy spotkań ani plików PDF.

Twoje zadanie:
- Rozpoznaj intencję pytania.
- Zwróć dokładnie jedną z odpowiedzi:  
  **WYWOŁAJ: AgentCheckFiles**  
  **WYWOŁAJ: AgentQ&A**  
  **WYWOŁAJ: AgentSummary**  
  **WYWOŁAJ: brak**

Zasady:
- Jeśli użytkownik chce tylko **zobaczyć listę plików** lub pyta o **dostępność plików**, zwróć: **AgentCheckFiles**
- Jeśli użytkownik pyta **co się wydarzyło podczas konkretnego spotkania**, zwróć: **AgentQ&A**
- Jeśli użytkownik prosi o **podsumowanie** spotkania (musi być w pytaniu słowo 'podsumowanie'), zwróć: **AgentSummary**
- Jeśli aktualne pytanie jest **kontynuacją wcześniejszego pytania (follow-up)** dotyczącego konkretnego spotkania (np. pytanie: „ile?”, „kto?”, „dlaczego?”, „czy tak było?”), również zwróć: **AgentQ&A**
- Jeśli pytanie nie dotyczy spotkań ani plików PDF, zwróć: **brak**

Uwaga: historia rozmowy będzie dołączona przed pytaniem użytkownika – wykorzystaj ją, aby rozpoznać follow-upy.

Przykłady:

Pytanie: „Czy są jakieś pliki PDF ze spotkań?”  
Odpowiedź: **WYWOŁAJ: AgentCheckFiles**

Pytanie: „Pokaż wszystkie dostępne pliki PDF”  
Odpowiedź: **WYWOŁAJ: AgentCheckFiles**

Pytanie: „Co omawiano na spotkaniu z 21 marca 2025 o godzinie 12:25?”  
Odpowiedź: **WYWOŁAJ: AgentQ&A**

Pytanie: „Czy były błędy krytyczne w spotkaniu z 21 marca 2025?”  
Odpowiedź: **WYWOŁAJ: AgentQ&A**

Pytanie: „Podsumuj spotkanie z 1 kwietnia 2025 o 14:00”  
Odpowiedź: **WYWOŁAJ: AgentSummary**

Pytanie: "Jaki jest cel zebrania w spotkaniu z 1 kwietnia 2025 o godzinie 14:00 ?"
Odpowiedź: **WYWOŁAJ: AgentQ&A**

Pytanie: „Ile było błędów?”  
Odpowiedź: **WYWOŁAJ: AgentQ&A** *(follow-up do wcześniejszego pytania o błędy)*

Pytanie: „Kto mówił o wzroście sprzedaży?”  
Odpowiedź: **WYWOŁAJ: AgentQ&A** *(follow-up do wcześniejszego pytania o przychody)*

Pytanie: „Czym jest sztuczna inteligencja?”  
Odpowiedź: **WYWOŁAJ: brak**

Pytanie: "Jaka jest aktualna pogoda w polsce ?"
odpowiedź: **WYWOŁAJ: brak**

""")

def prompt_check_files(available_files: list[str], user_question: str) -> SystemMessage:
    files_list = "\n".join(f"- {name}" for name in available_files)
    return f"""
Masz poniższą listę plików zapisanych w formacie "YYYY-MM-DD_HH-MM":

{files_list}

Pytanie użytkownika:
"{user_question}"

Odpowiedz zgodnie z tymi zasadami:
1. Jeśli pytanie dotyczy wszystkich plików — użyj wszystkich.
2. Jeśli pytanie dotyczy konkretnego pliku — użyj tylko tego pliku.
3. Jeśli pytanie dotyczy wielu plików z tą samą datą (YYYY-MM-DD) — pokaż wszystkie pasujące pliki z tą datą.

Jeśli plik, o który pyta użytkownik, nie istnieje — napisz, że nie znaleziono.

Twoja odpowiedź powinna być konkretna i zgodna z pytaniem.
"""

prompt_question_classifier = SystemMessage(content="""
Twoim zadaniem jest sprawdzić, czy pytanie użytkownika zawiera **konkretne odniesienie do daty** lub nazwę pliku (np. w formacie daty).

Jeśli pytanie zawiera:
- datę (np. "2025-10-21", "21 października 2025", "15-01-2023"),
lub
- dokładną nazwę pliku, lub numer dnia,

to odpowiedz: **filtr**

Jeśli pytanie NIE zawiera tych informacji (jest ogólne lub kontekstowe), odpowiedz: **brak**

Odpowiadasz tylko jednym słowem: **filtr** lub **brak**

Przykłady:

Pytanie: „Co omawiano na spotkaniu z 21 marca 2025?”  
Odpowiedź: filtr

Pytanie: „Czy ktoś wspomniał o błędach?”  
Odpowiedź: brak

Pytanie: „Spotkanie z 15-01-2023 – czy są z niego notatki?”  
Odpowiedź: filtr

Pytanie: „Pokaż mi listę tematów poruszanych przez ostatnie 3 tygodnie”  
Odpowiedź: brak

Pytanie: „Czy w pliku z 10-01-2025 była mowa o AI?”  
Odpowiedź: filtr
""")


prompt_retrieve_date = """Jesteś ekspertem od analizy pytań użytkowników, który ma za zadanie wydobyć datę z pytania.
    Jeśli pytanie zawiera datę w formacie:
    - yyyy-mm-dd lub yyyy-mm-dd hh:mm,
    - yyyy-mm-dd o godz. hh:mm,
    - yyyy-mm-dd o godzinie hh:mm,
    - dd-mm-yyyy lub dd-mm-yyyy hh:mm oraz dd-mm-yyyy o godzinie hh:mm,
    - dd-mm-yyyy lub dd-mm-yyyy hh:mm oraz dd-mm-yyyy o godz. hh:mm,
    - lub odnosi się do 'spotkania',
    to odpowiedz 'yyyy-mm-dd_hh-mm.pdf'.
    W przeciwnym razie odpowiedz 'nie podałeś daty spotkania'.
    Odpowiedź powinna być **tylko 'yyyy-mm-dd_hh-mm.pdf'**, bez żadnych dodatkowych informacji, kontekstu ani wyjaśnień.
    Przykłady:
    Pytanie: "Czy mogę prosić o podsumowanie spotkania z 21 października 2025 o godzinie 13:30?"
    Odpowiedź: 2025-10-21_13-30.pdf
    
    Pytanie: "Co było na spotkaniu 2024-12-11 o godz. 10:00?"
    Odpowiedź: 2024-12-11_10-00.pdf
    
    Pytanie: "Jakie decyzje podjęto na spotkaniu z 1 kwietnia 2025 o godzinie 14:00?"
    Odpowiedź: 2025-04-01_14-00.pdf
    """
    
def prompt_retrieve_date_for_single_file(answer_ai: str):
    return f"""Jesteś ekspertem od analizy pytań użytkowników,który ma za zadanie wydobyć nazwę pliku z pytania.
Masz listę nazw plików w formacie yyyy-mm-dd_hh-mm.
Na podstawie zapytania użytkownika, wybierz nazwę pliku, która najlepiej pasuje do jego prośby.
Zwróć tylko dokładną nazwę pasującego pliku.

Lista plików:
{answer_ai}


Oczekiwany wynik:
nazwa pliku w formacie yyyy-mm-dd_hh-mm.pdf, dopisz na końcu '.pdf'.
Jeśli nie ma pasującego pliku, odpowiedz 'nie mogę odpowiedzieć'. 
"""   
    
prompt_extract_date = """
Jesteś ekspertem językowym. Twoim zadaniem jest usunąć z pytania użytkownika wszelkie odniesienia do:

- słowa „spotkanie” (oraz jego odmian),
- dat (np. "1 kwietnia 2025", "03.02.2024", "2025-04-01"),
- godzin (np. "o godzinie 14:00", "godz. 12:25").

Jeśli pytanie zawiera słowo „spotkanie”, **usuń całą frazę od słowa 'spotkanie' aż do końca zdania lub znaku zapytania**.

Nie zmieniaj pozostałej treści pytania. Nie dodawaj żadnych dodatkowych komentarzy, tłumaczeń ani informacji. Zwróć tylko czyste, skrócone pytanie.

### Przykłady:

Input: "Jakie decyzje podjęto na spotkaniu 2025-04-01 o godzinie 14:00?"
Output: "Jakie decyzje podjęto?"

Input: "Czy klient zgłosił uwagi w spotkaniu z dnia 03.02.2024?"
Output: "Czy klient zgłosił uwagi?"

Input: "Czy były omawiane błędy krytyczne w spotkaniu z 21 marca 2025 o godzinie 12:25?"
Output: "Czy były omawiane błędy krytyczne?"

Input: "Jakie były tematy poruszane?"
Output: "Jakie były tematy poruszane?"
"""
    
prompt_retrieval_grade = SystemMessage(
        content="Jesteś ekspertem, który ocenia dokumenty pod kątem ich zgodności z pytaniem użytkownika. "
        "Jeśli dokumenty są zgodne z pytaniem, odpowiedz 'tak'. W przeciwnym razie odpowiedz 'nie'. "
        "Odpowiedź powinna być **tylko 'tak' lub 'nie'**, bez żadnych dodatkowych informacji, kontekstu ani wyjaśnień."
    )

prompt_refine_question = SystemMessage(
    content="""
    Jesteś ekspertem od optymalizacji zapytań użytkownika do systemów wyszukiwania semantycznego opartych na bazie wektorowej.

    Twoim zadaniem jest lekko udoskonalić pytanie użytkownika w języku **polskim**, aby:

    - było bardziej precyzyjne,
    - unikało niejasności,
    - zachowało pierwotne znaczenie,
    - zawierało słowa kluczowe istotne dla wyszukiwania semantycznego.

    Nie zmieniaj intencji pytania.
    Nie tłumacz pytania na inny język.
    Nie dodawaj informacji, których użytkownik nie podał.
    
    Zwróć tylko udoskonaloną wersję pytania w języku polskim.
    """
)

def prompt_check_user_agree(answer_user: str,question_from_ai: str) -> SystemMessage:
    return f"""
        Twoim zadaniem jest ocenić, czy użytkownik zgodził się na propozycję wyrażoną w pytaniu, np. „Czy chcesz zobaczyć zawartość pliku?”

Pytanie (od AI):
\"{question_from_ai}\"

Odpowiedź użytkownika:
\"{answer_user}\"

Zwróć tylko jedno słowo odpowiedzi: **tak** lub **nie** (małymi literami, bez kropek i bez dodatkowych wyjaśnień).

Przyjmij, że użytkownik wyraził zgodę jeśli:
- Odpowiedź zawiera słowa takie jak: „tak”, „oczywiście”, „jasne”, „pewnie”, „proszę”, „chętnie”, „możesz”, „pokaż”, „zobaczmy”,
- Użytkownik wskazuje na konkretny plik (np. „Pokaż plik 2024-05-01_12-00”, „Zobacz 01.05”),
- Odpowiedź jednoznacznie sugeruje chęć kontynuacji (np. „Wybieram pierwszy”, „Ten drugi plik”).

Jeśli użytkownik:
- Wyraźnie odmawia (np. „nie”, „nie chcę”, „nie teraz”, „nie pokazuj”, „dziękuję”),
- Jest niezdecydowany lub odpowiedź jest niejasna (np. „może”, „nie wiem”, „zastanowię się”),

→ wtedy odpowiedz **nie**.

Twoja odpowiedź musi być tylko jednym słowem: **tak** albo **nie**.
"""
    
def prompt_agent_router_files(ai_answer: str, user_question: str) -> SystemMessage:
    return SystemMessage(
        content=f"""
Masz poniżej odpowiedź AI oraz pytanie użytkownika.

Odpowiedź AI:
\"\"\"{ai_answer}\"\"\"

Pytanie użytkownika:
\"\"\"{user_question}\"\"\"

Format nazw plików to zawsze: YYYY-MM-DD_HH-MM (rok-miesiąc-dzień_punkt_godzina-minuta).

Na podstawie tych informacji oceń, czy pytanie dotyczy dokładnie jednego pliku czy wielu plików.

Zwróć:
- 's' jeśli chodzi o jeden plik,
- 'm' jeśli chodzi o wiele plików.
Odpowiedz TYLKO jedną literą: 's', 'm' (bez dodatkowych wyjaśnień, kropek czy innych znaków).
"""
    )