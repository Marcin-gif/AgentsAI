from typing import Any, Literal, Optional
from typing_extensions import TypedDict,Annotated,Sequence,List
from langchain_core.prompts import ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
import operator
from pandas import DataFrame
class AgentState(TypedDict):
    """AgentState służy jako struktura do przechowywania informacji o stanie agenta w trakcie całego procesu przetwarzania zapytania użytkownika."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    documents: Annotated[list, operator.add]
    use_filtr: str
    question: HumanMessage
    refined_question: HumanMessage
    refined_count: int
    proceed_to_generate: bool
    file_name: str
    files_names: list[str]
    specific_agent: str
    answer: str
    specific_file:str
    file_scope: str
    memory: dict[str, Any]

class AgentSupervisorState(TypedDict):
    """AgentSupervisorState służy jako struktura do przechowywania informacji o stanie agenta nadzorującego."""
    question: HumanMessage
    refined_question: HumanMessage
    specific_agent: str
    answer: str
    messages: Annotated[Sequence[BaseMessage], add_messages]

class AgentCheckFilesState(TypedDict):
    """AgentCheckFilesState służy jako struktura do przechowywania informacji o stanie agenta sprawdzającego pliki."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    documents: Annotated[list, operator.add]
    question: HumanMessage
    refined_question: HumanMessage
    file_scope: str
    file_name: str
    specific_file: str
    answer: str
    memory: dict[str, Any]
    
class AgentSummaryState(TypedDict):
    question: HumanMessage
    refined_question: HumanMessage
    messages: Annotated[Sequence[BaseMessage], add_messages]
    answer: str
    documents: Annotated[list, operator.add]
    file_name: str
    check_date_in_question: str
    
class AgentOCRState(TypedDict):
    question: HumanMessage
    refined_question: HumanMessage
    messages: Annotated[Sequence[BaseMessage], add_messages]
    answer: str
    image_path: str
    text: str
    extracted_data: Optional[dict]
    llm_validation_score: bool
    validation_count: int
    dataFrame: DataFrame
    dataFrame_info: dict[str, Any]
    is_Invoice: str
    db_pozycje: dict[str, Any]
    db_kupujacy: dict[str, Any]
    db_sprzedawca: dict[str, Any]
    db_odbiorca: dict[str, Any]
    db_faktura: dict[str, Any]
    db_podsumowanie: dict[str, Any]
    db_kwoty: dict[str, Any]
    user_response: dict
    ask_to_save: bool
    
class AgentSQLState(TypedDict):
    """AgentSQLState służy jako struktura do przechowywania informacji o stanie agenta SQL."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: HumanMessage
    refined_question: HumanMessage
    answer_sql: str
    answer: str
    
    
class AgentRouterFiles(BaseModel):
    """route to check if question is about one file or multiple files."""
    file: Literal["s","m"] = Field(
        description="Odpowiedz 's' jeśli pytanie dotyczy jednego pliku, jeśli pytanie dotyczy wielu plików to odpowiedz 'm'."
    )

class AgentCheckFiles(BaseModel):
    """Check if user agree to check file, or not."""
    file: Literal["t","n"] = Field(
        description="Odpowiedz 't' jeśli użytkownik zgadza się na sprawdzenie pliku, jeśli nie to odpowiedz 'n'."
    )

class AgentCheckDateQuestion(BaseModel):
    """Check if the question is about a specific date."""
    date: Literal["tak","nie"] = Field(
        description="Odpowiedz 'tak' jeśli w pytaniu znajduje się daty, jeśli nie to odpowiedz 'nie'."
    )

class AgentRouter(BaseModel):
    """Route a user query to the most relevant agent."""
    agent: Literal["AgentCheckFiles","AgentQ&A","AgentSummary","brak"] = Field(
        description="Odpowiedz 'AgentCheckFiles', 'AgentQ&A', 'AgentSummary', 'brak' w zależności od tego, który agent powinien zostać wywołany."
    )

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["filtr","brak"] = Field(
        description="Odpowiedz 'filtr' lub 'brak', zależnie od tego, czy pytanie zawiera słowo 'spotkanie' i datę z godziną jeśli tak odpowiedz 'filtr' jeśli nie zawiera to odpowiedz 'brak'."
    )

class DateQuery(BaseModel):
    """Extract a date from the user query."""
    date: str = Field(
        description="Odpowiedz 'yyyy-mm-dd_hh-mm' jeśli pytanie zawiera datę, w przeciwnym razie odpowiedz 'nie podałeś daty spotkania'."
    )
    
class GradeDocument(BaseModel):
    score: Literal["tak","nie"] = Field(
        description="Odpowiedz 'tak' lub 'nie', zależnie od tego, czy dokumenty są zgodne z pytaniem użytkownika."
    )
    
class ValidateJsonOutput(BaseModel):
    """Validate the JSON output from the agent."""
    valid: bool = Field(
        description="Odpowiedz 'true' jeśli JSON jest poprawny, w przeciwnym razie odpowiedz 'false'."
    )
    
class CheckIsInvoice(BaseModel):
    """Check if the question is about an invoice."""
    invoice: Literal["tak","nie"] = Field(
        description="Odpowiedz 'tak' jeśli pytanie dotyczy faktury, w przeciwnym razie odpowiedz 'nie'."
    )