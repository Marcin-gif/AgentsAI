from tools import name_files
from langchain.agents import initialize_agent, Tool
from langchain_ollama import ChatOllama
tools = [name_files]
model = ChatOllama(model="mistral", temperature=0.0)
model_wwith_tools=model.bind_tools(tools)
response = model_wwith_tools.invoke("Pokaż mi wszystkie pliki?")
print(response.content)
response = model_wwith_tools.invoke("Pokaż mi pliki z marca?")
print(response.content)