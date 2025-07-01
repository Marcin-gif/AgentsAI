from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import ollama
from PIL import Image
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
import re
import json
import pandas as pd
from state import AgentOCRState, ValidateJsonOutput,CheckIsInvoice
from langgraph.graph import StateGraph,END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from save_to_sql import insert_invoice_to_db

def load_image(state: AgentOCRState) -> AgentOCRState:
    """
    Load the image from the provided path in the state.
    """
    try:
        print("Load Image")
        image_path = state.get('image_path')
        if not image_path:
            raise ValueError("Brak ścieżki do obrazu w stanie.")

        # Tutaj wczytujemy obraz jako bajty, co jest wymagane przez ollama.chat z "images"
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # Ulepszony prompt dla modelu VLM (gemma3:12b)
        vlm_prompt = """
        Jesteś zaawansowanym asystentem do ekstrakcji danych z faktur.
        Przeanalizuj dokładnie dostarczony obraz faktury i wyodrębnij WSZYSTKIE dostępne informacje.
        Ustrukturyzuj wyodrębnione dane w postaci sekcji Markdown, używając nagłówków i list.
        Jeśli jakiejś kategorii lub pola nie ma na fakturze, po prostu ją pomiń lub zaznacz "brak".

        **Oczekiwane sekcje i pola:**

        **Dane Sprzedawcy:**
        * Nazwa: [Nazwa firmy]
        * Ulica: [Ulica i numer]
        * Kod pocztowy: [Kod pocztowy]
        * Miasto: [Miasto]
        * NIP: [Numer NIP]
        * Email: [Adres email, jeśli dostępny]
        * Telefon: [Numer telefonu, jeśli dostępny]
        * Numer konta bankowego: [Numer konta, jeśli dostępny]

        **Dane Kupującego:**
        * Nazwa: [Nazwa firmy]
        * Ulica: [Ulica i numer]
        * Kod pocztowy: [Kod pocztowy]
        * Miasto: [Miasto]
        * NIP: [Numer NIP]
        * Email: [Adres email, jeśli dostępny]
        * Telefon: [Numer telefonu, jeśli dostępny]

        **Dane Odbiorcy (jeśli różny od Kupującego):**
        * Nazwa: [Nazwa firmy]
        * Ulica: [Ulica i numer]
        * Kod pocztowy: [Kod pocztowy]
        * Miasto: [Miasto]
        * NIP: [Numer NIP]

        **Informacje o Fakturze:**
        * Typ dokumentu: [np. Faktura VAT, Paragony, Rachunek]
        * Numer faktury: [Pełny numer faktury]
        * Data wystawienia: [Data w formacie YYYY-MM-DD]
        * Data sprzedaży / Data wykonania usługi: [Data w formacie YYYY-MM-DD]
        * Sposób zapłaty: [np. Przelew, Gotówka, Karta]
        * Termin zapłaty: [Data w formacie YYYY-MM-DD]
        * Waluta: [np. PLN, EUR, USD]
        * Miejsce wystawienia: [Miasto]
        * P.O. Number (numer zamówienia): [Numer, jeśli dostępny]

        **Pozycje Faktury (każda pozycja jako oddzielny blok):**
        * LP: [Numer pozycji]
        * Nazwa towaru/usługi: [Nazwa]
        * Jednostka miary: [np. szt., kpl., godz.]
        * Ilość: [Liczba]
        * Cena jednostkowa netto: [Kwota]
        * Rabat (%): [Procent rabatu, np. 10%]
        * Wartość netto po rabacie: [Kwota]
        * Stawka VAT (%): [Procent VAT, np. 23%]
        * Kwota VAT: [Kwota]
        * Wartość brutto: [Kwota]

        **Podsumowanie Stawki VAT (jeśli są wyszczególnione):**
        * Stawka [X%]: Netto: [Kwota], VAT: [Kwota], Brutto: [Kwota]
        * Stawka [Y%]: Netto: [Kwota], VAT: [Kwota], Brutto: [Kwota]
        * ...

        **Osoba upoważniona do otrzyamiania faktury:**
        * Nazwa: [Imię i Nazwisko osoby, jeśli dostępna]
        
        **Osoba upoważniona do wystawiania faktury:**
        * Nazwa: [Imię i Nazwisko osoby, jeśli dostępna]
        
        **Całkowite Podsumowanie Faktury:**
        * Suma netto: [Całkowita kwota netto]
        * Suma VAT: [Całkowita kwota VAT]
        * Suma brutto do zapłaty: [Całkowita kwota brutto]
        * Słownie: [Kwota słownie]
        * Zapłacono: [Kwota, jeśli faktura została częściowo/całkowicie opłacona]
        * Pozostało do zapłaty: [Kwota]

        **Dodatkowe Informacje:**
        * Uwagi: [Wszelkie ogólne uwagi, np. "proszę potwierdzić odbiór przesyłki"]
        * Podpisy: [np. Sprzedawca, Nabywca]
        * Inne: [Wszelkie inne istotne dane, które nie pasują do powyższych kategorii]

        Upewnij się, że wszystkie liczby są wyodrębnione z kropką dziesiętną jako separatorem.
        """
       

        response = ollama.chat(
        model="qwen2.5vl:7b",
        messages=[{
            "role": "user",
            "content": f"Wyodrębnij wszystkie dostępne informacje z faktury na obrazie. {vlm_prompt}",
            "images": [image_bytes]
            }]
        )

        cleaned_text = response['message']['content'].strip()
        print("Extracted Text:")
        print(cleaned_text)
        state['text'] = cleaned_text
        return state
    except Exception as e:
        print(f"Error loading image: {e}")
        return state

def is_invoice(state: AgentOCRState) -> AgentOCRState:
    """
    Check if the extracted text contains invoice-related keywords.
    """
    print("Checking if the text is an invoice...")
    try:
        text = state.get('text','')
        if not text:
            print("No text extracted from the image.")
            return state
        prompt="""
            Na podstawie tekstu z obrazu, określ, czy dany dokument to faktura.

            W oparciu o analizę tekstu, odpowiedz "tak" lub "nie".

            **tekst:**
            {text}

            Odpowiedź powinna być jednym słowem: "tak" lub "nie".
        """
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system",prompt),
            HumanMessage(content="{text}")
        ])
        
        llm = ChatOllama(model="deepseek-r1")
        structured_llm_router = llm.with_structured_output(CheckIsInvoice)
        
        chain = prompt_template | structured_llm_router
        
        result = chain.invoke({"text": text})
        print("Is invoice result:", result.invoice)
        state['is_Invoice'] = result.invoice
        return state
        
    except Exception as e:
        print(f"Error checking if text is invoice: {e}")
        return state

def check_is_invoice(state: AgentOCRState) -> str:
    """
    Check if the extracted text is an invoice.
    """
    print("Checking if the text is an invoice...")
    if state.get('is_Invoice') == 'tak':
        return "generate_json_output"
    else:
        return "cannot_answer"

def generate_json_output(state: AgentOCRState) -> AgentOCRState:
    """
    Extracted text from the state and generate a structured JSON output using 
    model llama3:8b.
    """
    try:
        print("Generating JSON output from extracted text...")
        system_prompt = """
           Jesteś agentem AI, którego zadaniem jest wyodrębnienie danych z faktury i zwrócenie ich w ściśle określonym formacie JSON. 

### Struktura JSON musi zawierać pola:

{{
  "sprzedawca": {{
    "nazwa": ...,
    "ulica": ...,
    "kod_pocztowy": ...,
    "miasto": ...,
    "nip": ...,
    "email": ...       // jeśli brak email, wpisz null
  }},
  "kupujacy": {{
    "nazwa": ...,
    "ulica": ...,
    "kod_pocztowy": ...,
    "miasto": ...,
    "nip": ...,
    "email": ...       // jeśli brak email, wpisz null
  }},
  "odbiorca": {{
    "nazwa": ...,
    "ulica": ...,
    "kod_pocztowy": ...,
    "miasto": ...,
    "nip": ...,
    "email": ...       // jeśli brak email, wpisz null
  }},
  "faktura": {{
    "numer_faktury": ...,
    "data_wystawienia": ...,
    "data_sprzedazy": ...,
    "sposob_zaplata": ...,
    "termin_zaplata": ...,
    "po_number": ...
  }},
  "pozycje": [
    {{
      "lp": ...,
      "nazwa_towaru_uslugi": ...,
      "rabat_procent": ...,
      "ilosc": ...,
      "jednostka_miary": ...,
      "cena_netto_przed_rabatem": ...,
      "cena_netto_po_rabacie": ...,
      "wartosc_netto_pozycji": ...,
      "stawka_vat_procent": ...,
      "kwota_vat_pozycji": ...,
      "wartosc_brutto_pozycji": ...
    }}
  ],
  "podsumowanie": {{
    "wartosc_netto_sumaryczna": ...,
    "stawka_vat_sumaryczna_procent": ...,
    "wartosc_vat_sumaryczna": ...,
    "wartosc_brutto_sumaryczna": ...,
    "waluta": ...
  }},
  "zaplacono": ...,
  "pozostalo_do_zaplaty": ...,
  "slownie": ...,
  "kwota_do_zaplaty_koncowa": ...,
  "uwagi": ...
}}

---

**WAŻNE:**

- Każde pole musi być obecne. 
- Jeśli w tekście faktury brakuje jakiejś wartości, wstaw `null`.
- Pole `email` szukaj szczególnie w sekcjach sprzedawca, kupujący i odbiorca.
- Jeśli nie znajdziesz adresu email, wpisz null.
- Format JSON musi być poprawny i parsowalny.

---

Input faktury:
{response}


        """

        text="""
            Extracted Text:
        Oto wyciągnięte dane z obrazu faktury:

        **Dane Sprzedawcy:**

        *   Nazwa: Firma ABC
        *   Adres: ul. Słowackiego 2
        *   Kod pocztowy: 11-455
        *   Miasto: Katowice
        *   NIP: 8731407921

        **Dane Kupującego:**

        *   Nazwa: Produkcja Sp. z o.o.
        *   Adres: ul. Nowogrodzka 5
        *   Kod pocztowy: 00-957
        *   Miasto: Warszawa
        *   NIP: 9372424178

        **Dane Odbiorcy:**
        *   Nazwa: Handel S.C.
        *   Adres: ul. prosta 8
        *   Kod pocztowy: 00-121
        *   Miasto: Warszawa
        *   NIP: 5111625852

        **Informacje o Fakturze:**

        *   Data wystawienia: 2019-08-28
        *   Data sprzedaży: 2019-08-28
        *   Numer faktury: FV 1/09/2019
        *   Sposób zapłaty: Przelew
        *   Termin zapłaty: 2019-10-05

        **Pozycje Faktury:**

        *   **LP 1:**
            *   Nazwa towaru/usługi: Oprogramowanie
            *   Rabaty: 20%
            *   Ilość: 1 szt.
            *   Cena netto: 16200.00
            *   Cena netto po rabacie: 12960.00
            *   Wartość netto: 12960.00
            *   Stawka VAT: 23%
            *   Wartość VAT: 2983.10
            *   Wartość brutto: 15943.10
        *   **LP 2:**
            *   Nazwa towaru/usługi: Dostawa
            *   Rabaty: 0%
            *   Ilość: 1 szt.
            *   Cena netto: 10.00
            *   Cena netto po rabacie: 10.00
            *   Wartość netto: 10.00
            *   Stawka VAT: 23%
            *   Wartość VAT: 2.30
            *   Wartość brutto: 12.30

        **Podsumowanie:**

        *   Wartość netto: 12970.00 PLN
        *   Stawka VAT: 23%
        *   Wartość VAT: 2983.10 PLN
        *   Wartość brutto: 15953.10 PLN

        **Dodatkowe informacje:**

        *   Zapłacono: 0.00 PLN
        *   Pozostało do zapłaty: 15953.10 PLN
        *   Słownie: pięćnaście tysięcy dziewięćset pięćdziesiąt trzy i 10/100 PLN
        *   Kwota do zapłaty: 15953.10 PLN
        *   Uwagi: proszę potwierdzić odbiór przesyłki
        *   Odbiorca: Handel S.C.



        Proszę dać znać, jeśli potrzebujesz więcej informacji.

        """
        transform_text = state.get('text', '')
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            # Tutaj dodajemy HumanMessage, która zawiera zmienną {response}
            HumanMessage(content="{response}")
        ])

        llm = ChatOllama(model="llama3:8b")

        chain = (prompt | llm | StrOutputParser())

        result = chain.invoke({"response": transform_text})
        print(result)
        state['extracted_data'] = result
        return state
    except Exception as e:
        print(f"Error generating JSON output: {e}")
        return state
    
def validate_json_output(state: AgentOCRState) -> AgentOCRState:
    """
        Validate the extracted JSON data using the model llama3:8b.
    """
    print("Validating JSON output...")
    text = state.get('text')
    extracted_data = state.get('extracted_data')
    if not text or not extracted_data:
        print("Text or extracted data is missing.")
        return state
    try:
        system_prompt = """
            Jesteś zaawansowanym asystentem do walidacji danych z faktur.
            Twoim zadaniem jest sprawdzenie, czy podany obiekt JSON jest poprawny i zgodny z wymaganym schematem Pydantic.

            **Kluczowe zasady walidacji:**

            1.  **Poprawność JSON:** Sprawdź, czy obiekt JSON jest poprawnie sformatowany.
            2.  **Zgodność ze schematem:** Upewnij się, że wszystkie wymagane pola są obecne i mają odpowiednie typy danych.
            3.  **Brakujące dane:** Jeśli jakieś pole jest nieobecne, powinno być ustawione na `null`.
            4.  **Format dat:** Sprawdź, czy daty są w formacie `YYYY-MM-DD`.
            5.  **Typy danych:** Upewnij się, że liczby są typu `float` z kropką dziesiętną.

            Input text:
            {text}

            Input JSON:
            {extracted_data}
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            HumanMessage(content="{text}"),
            HumanMessage(content="{extracted_data}")
        ])

        llm = ChatOllama(model="llama3:8b")
        structured_llm_router = llm.with_structured_output(ValidateJsonOutput)
        chain = (prompt | structured_llm_router)

        result = chain.invoke({"text": text, "extracted_data": extracted_data})
        print(result)
        state['llm_validation_score'] = result.valid
        state['validation_count'] = state.get('validation_count', 0) + 1
        return state
    except Exception as e:
        print(f"Error validating JSON output: {e}")
        state['llm_validation_score'] = False
        
        return state

def check_validation_score(state: AgentOCRState) -> str:
    """
    Check the validation score and decide whether to continue or stop.
    """
    print("Checking validation score...")
    if state.get('llm_validation_score', False):
        return "generate_dataFrame"
    elif state.get('validation_count', 0) >= 2:
        return "cannot_answer"
    else:
        return "generate_json_output"
    

def generate_dataFrame(state: AgentOCRState) -> AgentOCRState:
    """
    Generate the final answer based on the extracted data.
    """
    print("Generating DataFrame from extracted data...")
    result = state.get('extracted_data', '')
    if not result:
        print("No extracted data available.")
        return state
    
    # Debug: Print the extracted_data to see its format
    print("DEBUG - Extracted data:")
    print(repr(result))
    
    # Try multiple regex patterns to find JSON data
    json_patterns = [
        r"```json\n(.*?)\n```",  # ```json ... ```
        r"```\n(.*?)\n```",      # ``` ... ```
        r"```(.*?)```",          # ```...``` (no newlines)
        r"\{.*\}",               # Direct JSON object
    ]
    
    parsed_data = None
    
    for pattern in json_patterns:
        json_match = re.search(pattern, result, re.DOTALL)
        if json_match:
            receipt_data = json_match.group(1) if pattern != r"\{.*\}" else json_match.group(0)
            try:
                parsed_data = json.loads(receipt_data)
                print(f"Successfully parsed JSON with pattern: {pattern}")
                break
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                continue

    if not parsed_data:
        print("No JSON data found.")
        return state

    receipt_dict = parsed_data
    print("Parsed receipt dictionary:")
    print(json.dumps(receipt_dict, indent=2, ensure_ascii=False))

    # Store each section in state
    for section in ['sprzedawca', 'kupujacy', 'odbiorca', 'faktura', 'podsumowanie']:
        state[f'db_{section}'] = receipt_dict.get(section, {})
        print(state[f'db_{section}'])

    # Handle items (pozycje)
    try:
        pozycje = receipt_dict.get("pozycje", [])
        items_df = pd.DataFrame(pozycje)
        print("Items DataFrame:")
        print(items_df)

        state['db_pozycje'] = items_df.to_dict(orient='records')
        state['dataFrame_info'] = {
            'columns': items_df.columns.tolist(),
            'shape': list(items_df.shape),
            'dtypes': {col: str(dtype) for col, dtype in items_df.dtypes.items()}
        }

    except Exception as e:
        print(f"Error creating DataFrame for 'pozycje': {e}")
        state['db_pozycje'] = []

    # Store payment and remarks info
    state['db_kwoty'] = {
        'kwota_do_zaplaty_koncowa': receipt_dict.get('kwota_do_zaplaty_koncowa'),
        'zaplacono': receipt_dict.get('zaplacono'),
        'pozostalo_do_zaplaty': receipt_dict.get('pozostalo_do_zaplaty'),
        'slownie': receipt_dict.get('slownie'),
        'uwagi': receipt_dict.get('uwagi'),
    }

    print("\n--- FINAL STATE KEYS ---")
    for key in state.keys():
        if key.startswith("db_"):
            print(f"{key}: {type(state[key])}")

    return state


def generate_answer(state: AgentOCRState) -> AgentOCRState:
    """
    Generate the final answer based on the extracted data.
    """
    print("Generating final answer based on extracted data...")
    extracted_data = state.get('extracted_data', '')
    question = state.get('refined_question', '')
    if not extracted_data or not question:
        print("No extracted data or question available.")
        return state
    
    system_prompt = """
        Jesteś zaawansowanym asystentem do generowania odpowiedzi na podstawie danych z faktur.
        Twoim zadaniem jest wygenerowanie odpowiedzi na pytanie użytkownika, bazując na podanych danych.

        **Kluczowe zasady generowania odpowiedzi:**

        1. **Użyj wszystkich dostępnych danych:** Wykorzystaj wszystkie informacje zawarte w danych.
        2. **Odpowiedź powinna być zwięzła i precyzyjna:** Skup się na najważniejszych informacjach.
        3. **Format odpowiedzi:** Odpowiedź powinna być w formie tekstu, nie JSON.

        Pytanie użytkownika: {question}

        Dane z faktury:
        {extracted_data}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Wygeneruj odpowiedź na podstawie powyższych danych.")
    ])

    llm = ChatOllama(model="llama3:8b")
    chain = (prompt | llm | StrOutputParser())

    result = chain.invoke({"extracted_data": extracted_data, "question": question})
    print("Generated answer:")
    print(result)
    state['answer'] = result
    return state

def cannot_answer(state: AgentOCRState) -> AgentOCRState:
    """
    Handle the case where the agent cannot answer the question.
    """
    print("Cannot answer the question.")
    print("Entering cannot answer")
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    state["messages"].append(
        HumanMessage(
            content="Przepraszam, ale nie mogę znaleźć informacji których szukasz."
        )
    )
    print(HumanMessage(content="Przepraszam, ale nie mogę znaleźć informacji których szukasz."))
    state["answer"] = HumanMessage(
        content="Przepraszam, ale nie mogę znaleźć informacji których szukasz."
    )
    return state

def save_to_db(state: AgentOCRState) -> AgentOCRState:
    print("Saving invoice data to database...")
    invoice_json = {
        "sprzedawca": state.get("db_sprzedawca", {}),
        "kupujacy": state.get("db_kupujacy", {}),
        "odbiorca": state.get("db_odbiorca", {}),
        "faktura": state.get("db_faktura", {}),
        "podsumowanie": state.get("db_podsumowanie", {}),
        "pozycje": state.get("db_pozycje", []),
        "zaplacono": state['db_kwoty'].get('zaplacono', 0.0),
        "pozostalo_do_zaplaty": state['db_kwoty'].get('pozostalo_do_zaplaty', 0.0),
        "kwota_do_zaplaty_koncowa": state['db_kwoty'].get('kwota_do_zaplaty_koncowa', 0.0),
        "slownie": state['db_kwoty'].get('slownie', ""),
        "uwagi": state['db_kwoty'].get('uwagi', "")
    }
    insert_invoice_to_db(invoice_json)
    print("Invoice data saved to DB.")
    return state

# def human_review(state: AgentOCRState):
#     return interrupt({"pozycje": state.get("db_pozycje", []),"ask_to_save":True})

# def apply_user_input(state: AgentOCRState):
#     feedback = state.get("user_response", {})
#     if "pozycje" in feedback:
#         state["db_pozycje"] = feedback["pozycje"]
#         print("Updated 'pozycje' in state with user input.")
#     return state

# def save_data(state: AgentOCRState):
#     """
#     Save the final state to a file or database.
#     This is a placeholder function for saving the data.
#     """
#     if state.get("save_to_db"): # zapisujesz wszystkie pola db_*
#         state["answer"] = "✅ Dane zostały zapisane do bazy danych."
#     else:
#         state["answer"] = "ℹ️ Dane nie zostały zapisane."

#     return state

memorySaver = MemorySaver()

workflow = StateGraph(AgentOCRState)
workflow.add_node(
    "load_image",
    load_image
)
workflow.add_node(
    "is_invoice",
    is_invoice
)
workflow.add_node(
    "generate_json_output",
    generate_json_output
)
workflow.add_node(
    "validate_json_output",
    validate_json_output
)

workflow.add_node(
    "generate_dataFrame",
    generate_dataFrame
)
workflow.add_node(
    "generate_answer",
    generate_answer
)
workflow.add_node(
    "cannot_answer",
    cannot_answer
)
workflow.add_node("save_to_db", save_to_db)
# workflow.add_node("human_review", human_review)
# workflow.add_node("apply_user_input", apply_user_input)
# workflow.add_node("save_data", save_data)

workflow.add_edge("load_image", "is_invoice") 
workflow.add_conditional_edges(
    "is_invoice",
    check_is_invoice,
    {
        "generate_json_output": "generate_json_output",
        "cannot_answer": "cannot_answer"
    }
)
workflow.add_edge("generate_json_output", "validate_json_output")
workflow.add_conditional_edges(
    "validate_json_output",
    check_validation_score,
    {
        "generate_dataFrame": "generate_dataFrame",
        "generate_json_output": "generate_json_output",
        "cannot_answer": "cannot_answer"
    }
)
workflow.add_edge("generate_dataFrame", "generate_answer")
workflow.add_edge("generate_answer", "save_to_db")
workflow.add_edge("save_to_db",END)
# workflow.add_edge("generate_answer", "human_review")
# workflow.add_edge("human_review", "apply_user_input")
# workflow.add_edge("apply_user_input", "save_data")
# workflow.add_edge("save_data", END)
workflow.add_edge("cannot_answer", END)
workflow.set_entry_point("load_image")
graph_agent_ocr = workflow.compile(checkpointer=memorySaver)
graph_agent_ocr.get_graph().draw_mermaid_png(output_file_path="AgentOCRGraph.png")

