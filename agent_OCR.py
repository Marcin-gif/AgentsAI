from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import ollama
import pytesseract
from PIL import Image
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
import re
import json
import pandas as pd
from state import AgentOCRState, ValidateJsonOutput
from langgraph.graph import StateGraph,END
from langgraph.checkpoint.memory import MemorySaver

def load_image(state: AgentOCRState) -> AgentOCRState:
    """
    Load the image from the provided path in the state.
    """
    try:
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


def generate_json_output(state: AgentOCRState) -> AgentOCRState:
    """
    Extracted text from the state and generate a structured JSON output using 
    model llama3:8b.
    """
    try:
        system_prompt = """
            Jesteś zaawansowanym asystentem do ekstrakcji danych z faktur.
            Twoim zadaniem jest precyzyjne wyodrębnienie wszystkich dostępnych informacji z podanego tekstu faktury
            i przekształcenie ich w ustrukturyzowany obiekt JSON, ściśle zgodny z poniższym schematem Pydantic.

            **Kluczowe zasady ekstrakcji i mapowania:**

            1.  **Format JSON:** Zawsze zwracaj tylko kompletny i poprawny obiekt JSON.
            2.  **Pełne mapowanie pól:**
                * **Sprzedawca, Kupujący, Odbiorca:** Traktuj te podmioty jako obiekty typu `DaneFirmy`.
                    * Dla każdego z nich (sprzedawca, kupujacy, odbiorca), mapuj dostępne pola takie jak:
                        * `Nazwa` -> `nazwa`,
                        * `Ulica` -> `ulica`,
                        * `Kod pocztowy` -> `kod_pocztowy`,
                        * `Miasto` -> `miasto`,
                        * `NIP` -> `nip`,
                        * `Email` -> `email`.
                    * **Bardzo Ważne dla Odbiorcy:** Nawet jeśli dla "Odbiorca" w tekście podana jest tylko `Nazwa` ("Handel S.C."), model **MUSI** utworzyć pełny obiekt `odbiorca` typu `DaneFirmy`. Pola takie jak `adres`, `kod_pocztowy`, `miasto`, `nip`, `email`, które nie zostały znalezione w tekście dla odbiorcy, **muszą zostać ustawione na `null`**. Nie pomijaj obiektu `odbiorca` tylko dlatego, że jest niekompletny.
                * **Faktura (obiekt `faktura`):** Zbierz wszystkie ogólne informacje o fakturze w zagnieżdżonym obiekcie `faktura`:
                    * `Numer faktury` -> `numer_faktury`
                    * `Data wystawienia` -> `data_wystawienia` (format `YYYY-MM-DD`)
                    * `Data sprzedaży` -> `data_sprzedazy` (format `YYYY-MM-DD`)
                    * `Sposób zapłaty` -> `sposob_zaplata`
                    * `Termin zapłaty` -> `termin_zaplata` (format `YYYY-MM-DD`)
                    * `P.O. Number` (jeśli wystąpi) -> `po_number` (w tym przypadku brak, więc `null`)
                * **Pozycje Faktury (lista `pozycje`):** Dla każdej pozycji:
                    * `LP X` -> `lp` (jako liczba całkowita, np. 1, 2)
                    * `Nazwa towaru/usługi` -> `nazwa_towaru_uslugi`
                    * `Rabaty` -> `rabat_procent` (jako float, np. 20.0 dla 20%)
                    * `Ilość` -> `ilosc` (jako float)
                    * `Jednostka miary` -> `jednostka_miary` (np. "szt.")
                    * `Cena netto` -> `cena_netto_przed_rabatem` (jako float)
                    * `Cena netto po rabacie` -> `cena_netto_po_rabacie` (jako float)
                    * `Wartość netto` (dla pozycji) -> `wartosc_netto_pozycji` (jako float)
                    * `Stawka VAT` -> `stawka_vat_procent` (jako float, np. 23.0 dla 23%)
                    * `Wartość VAT` (dla pozycji) -> `kwota_vat_pozycji` (jako float)
                    * `Wartość brutto` (dla pozycji) -> `wartosc_brutto_pozycji` (jako float)
                * **Podsumowanie (obiekt `podsumowanie`):** Zbierz wszystkie ogólne sumy w zagnieżdżonym obiekcie `podsumowanie`:
                    * `Wartość netto` -> `wartosc_netto_sumaryczna` (jako float)
                    * `Stawka VAT` (ogólna) -> `stawka_vat_sumaryczna_procent` (jako float)
                    * `Wartość VAT` (ogólna) -> `wartosc_vat_sumaryczna` (jako float)
                    * `Wartość brutto` -> `wartosc_brutto_sumaryczna` (jako float)
                    * `PLN` -> `waluta` (domyślnie "PLN")
                * **Dodatkowe informacje (pola na najwyższym poziomie):**
                    * `Zapłacono` -> `zaplacono` (jako float)
                    * `Pozostało do zapłaty` -> `pozostalo_do_zaplaty` (jako float)
                    * `Słownie` -> `slownie`
                    * `Kwota do zapłaty` -> `kwota_do_zaplaty_koncowa` (jako float)
                    * `Uwagi: proszę potwierdzić odbiór przesyłki` -> `uwagi` (tylko tekst "proszę potwierdzić odbiór przesyłki"). **NIE dołączaj innych wartości z sekcji "Dodatkowe informacje" do pola `uwagi`.**

            3.  **Typy Danych i Separatory:** Konwertuj wszystkie wartości liczbowe na typ `float` z kropką dziesiętną. Daty w formacie `YYYY-MM-DD`.
            4.  **Brakujące Dane:** Jeśli dane pole nie występuje w tekście, ustaw jego wartość na `null`. Nie pomijaj całych obiektów/pól tylko dlatego, że są niekompletne, jeśli schemat tego wymaga.
            5.  **Priorytet Sum:** W przypadku rozbieżności między sumami pozycji a sumami podanymi w sekcji "Podsumowanie", zawsze preferuj wartości z "Podsumowania".

            Input text:
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
            if pattern == r"\{.*\}":
                receipt_data = json_match.group(0)  # Full match for direct JSON
            else:
                receipt_data = json_match.group(1)  # First capture group
            
            try:
                parsed_data = json.loads(receipt_data)
                print(f"Successfully parsed JSON with pattern: {pattern}")
                break
            except json.JSONDecodeError:
                continue
    
    if parsed_data:
        json_data = json.dumps(parsed_data, indent=2)
        print("Parsed JSON data:")
        print(json_data)
        
        # Convert to DataFrame and update state
        receipt_dict = parsed_data
        print("Receipt dictionary:")
        print(receipt_dict)
        
        try:
            items_df = pd.DataFrame(receipt_dict)
            print("DataFrame:")
            print(items_df)
            
            # Store serializable data
            state['dataFrame'] = items_df.to_dict(orient='records')
            state['dataFrame_info'] = {
                'columns': items_df.columns.tolist(),
                'shape': list(items_df.shape),  # Convert tuple to list
                'dtypes': {col: str(dtype) for col, dtype in items_df.dtypes.items()}  # Convert dtypes to strings
            }
            
        except Exception as e:
            print(f"Error creating DataFrame: {e}")
            # If direct conversion fails, try different approaches
            if isinstance(receipt_dict, dict):
                if 'items' in receipt_dict:
                    items_df = pd.DataFrame(receipt_dict['items'])
                elif 'pozycje' in receipt_dict:  # Polish word for 'items'
                    items_df = pd.DataFrame(receipt_dict['pozycje'])
                elif any(isinstance(v, list) for v in receipt_dict.values()):
                    # Find the first list value and use it
                    for key, value in receipt_dict.items():
                        if isinstance(value, list):
                            items_df = pd.DataFrame(value)
                            break
                else:
                    # Convert single dict to single-row DataFrame
                    items_df = pd.DataFrame([receipt_dict])
                
                print("DataFrame (alternative method):")
                print(items_df)
                
                # Store serializable data
                state['dataFrame'] = items_df.to_dict(orient='records')
                state['dataFrame_info'] = {
                    'columns': items_df.columns.tolist(),
                    'shape': list(items_df.shape),  # Convert tuple to list
                    'dtypes': {col: str(dtype) for col, dtype in items_df.dtypes.items()}  # Convert dtypes to strings
                }
    else:
        print("No JSON data found in the response with any pattern.")
        return state

    return state


def generate_answer(state: AgentOCRState) -> AgentOCRState:
    """
    Generate the final answer based on the extracted data.
    """
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

memorySaver = MemorySaver()

workflow = StateGraph(AgentOCRState)
workflow.add_node(
    "load_image",
    load_image
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

workflow.add_edge("load_image", "generate_json_output") 
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
workflow.add_edge("generate_answer", END)
workflow.add_edge("cannot_answer", END)
workflow.set_entry_point("load_image")
graph_agent_ocr = workflow.compile(checkpointer=memorySaver)
graph_agent_ocr.get_graph().draw_mermaid_png(output_file_path="AgentOCRGraph.png")

result = graph_agent_ocr.invoke(
    {
        "image_path": "invoice.jpg",
        "refined_question": HumanMessage(content="Napisz mi wszystko o towarach?"),
        "messages": []
    },
    config={"configurable": {"thread_id": 1}}
)

print("Final Result: ", result.get("answer", "No answer generated."))