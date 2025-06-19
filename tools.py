from langchain.tools import tool
from state import AgentState
from pathlib import Path
from langchain_core.messages import HumanMessage
def name_files() -> list[str]:
    """Get all pdf files names in current directory
        args:none
        return: list of pdf files names
    """
    pdf_files = Path("./").glob("*.pdf")
    files_name = [f.stem for f in pdf_files]
    return files_name


def init_agent_state(question: HumanMessage) -> AgentState:
    return {
        "question": question,
        "refined_question": question,
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