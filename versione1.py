import os
import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import AIMessage
from langgraph.prebuilt import tools_condition, ToolNode
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver  # Importiamo MemorySaver per la memoria

# === SETUP ===
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)

# === TOOLS ===
@tool
def search_web(topic: str) -> dict:
    """Cerca articoli sportivi recenti su ESPN o Gazzetta in base a un argomento."""
    url = f"https://html.duckduckgo.com/html/?q={topic}+site:gazzetta.it+OR+site:espn.com"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    results = []
    for link in soup.select(".result__a")[:5]:
        href = link.get("href")
        if href:
            results.append(href)
    return {"urls": results}

@tool
def evaluate_source(url: str) -> str:
    """Valuta l'affidabilità di una fonte per un blog sportivo."""
    messages = [
        SystemMessage(content="Agisci come valutatore di fonti per un blog sportivo."),
        HumanMessage(content=f"Sono un blogger e sto scrivendo un articolo sportivo.\n"
                             f"Sto considerando questa fonte: {url}\n"
                             f"Valuta questa fonte e rispondi in JSON:\n"
                             '{"score": int (1-10), "comment": "breve motivazione"}')
    ]
    return llm.invoke(messages).content

@tool
def generate_article(prompt: str) -> str:
    """Genera un articolo sportivo basato su un prompt dell'utente."""
    messages = [
        SystemMessage(content="Agisci come esperto sportivo e scrivi un articolo in base al prompt dato."),
        HumanMessage(content=f"Prompt dell'articolo: {prompt}")
    ]
    return llm.invoke(messages).content

@tool
def check_fact(content: str, url: str, evaluation: str) -> str:
    """Conferma o smentisce il contenuto di un articolo usando una fonte e la sua valutazione."""
    messages = [
        SystemMessage(content="Sei un fact-checker per articoli sportivi."),
        HumanMessage(content=f"Sto scrivendo questo articolo:\n\n{content}\n\n"
                             f"La fonte è: {url}\n"
                             f"Valutazione della fonte: {evaluation}\n"
                             "La fonte conferma il contenuto? Rispondi spiegando eventuali discrepanze o conferme.")
    ]
    return llm.invoke(messages).content

@tool
def generate_report(checked_sources: str) -> str:
    """Genera un report sull'affidabilità dell'articolo e suggerisce miglioramenti."""
    messages = [
        SystemMessage(content="Agisci come editor e ottimizzatore di contenuti per blog sportivi."),
        HumanMessage(content="Ecco le fonti analizzate e i risultati del fact-checking:\n"
                             f"{checked_sources}\n\n"
                             "Sulla base di queste, genera un report che:\n"
                             "- Valuta l'affidabilità generale del mio articolo\n"
                             "- Indica se ci sono elementi da correggere\n"
                             "- Suggerisce miglioramenti stilistici o di contenuto")
    ]
    return llm.invoke(messages).content


# === LLM WITH TOOLS ===

tools = [search_web, evaluate_source, generate_article, check_fact, generate_report]
llm_with_tools = llm.bind_tools(tools=tools, tool_choice="auto")

# === ASSISTANT NODE ===

sys_msg = SystemMessage(content="""
Sei un assistente per un blog sportivo.

🛑 Non puoi rispondere direttamente all'utente.  
✅ Il tuo unico compito è scegliere quale tool usare.  
🔧 Usa sempre e solo i tools disponibili, altrimenti restituisci un errore (via tool_call fittizia se necessario).  
🚫 Se nessun tool è adatto, **non rispondere**.

Tools disponibili:
- `search_web`: cerca articoli sportivi recenti
- `evaluate_source`: valuta l'affidabilità di una fonte
- `generate_article`: genera articoli sportivi da un prompt
- `check_fact`: verifica fatti basandosi su una fonte e valutazione
- `generate_report`: crea un report editoriale sulle fonti

Parla solo di sport.
""")

def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# === MEMORY SETUP ===
memory = MemorySaver()  # Inizializza la memoria

# === GRAPH ===
builder = StateGraph(MessagesState)

# Nodi
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Edge
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
builder.add_edge("assistant", END)

# Configurazione per la memoria
config = {
    "configurable": {
        "thread_id": "1",  # Un thread_id unico per il flusso
        "checkpoint_ns": "article_checkpoints",  # Namespace per il checkpoint
        "checkpoint_id": "article_1"  # ID del checkpoint
    }
}

# Compilazione del grafico con la memoria
react_graph_memory = builder.compile(checkpointer=memory)

# === ZONA DI PROVA ===

from langchain_core.messages import HumanMessage

# Prima interazione
messages = [HumanMessage(content="Scrivimi un articolo su Tadej Pogacar massimo di 200 caratteri")]
output = react_graph_memory.invoke({
    "messages": messages
}, config)  # Passiamo anche la configurazione

for msg in output["messages"]:
    print(f"{msg.type.upper()} :\n{msg.content}\n{'-'*50}")

tool_invoked = False

print("\n=== TOOL INVOCATI DURANTE IL FLUSSO ===")
for msg in output["messages"]:
    if isinstance(msg, AIMessage) and msg.tool_calls:
        tool_invoked = True
        for call in msg.tool_calls:
            print(f"- Tool: {call['name']}")
            print(f"  Args: {call['args']}")
            print("-" * 40)

if not tool_invoked:
    print("Nessun tool è stato invocato.")

# Seconda interazione con memoria
messages = [HumanMessage(content="me lo potresti riformulare?.")]
output = react_graph_memory.invoke({
    "messages": messages
}, config)  # Passiamo anche la configurazione

for msg in output["messages"]:
    print(f"{msg.type.upper()} :\n{msg.content}\n{'-'*50}")

tool_invoked = False

print("\n=== TOOL INVOCATI DURANTE IL FLUSSO ===")
for msg in output["messages"]:
    if isinstance(msg, AIMessage) and msg.tool_calls:
        tool_invoked = True
        for call in msg.tool_calls:
            print(f"- Tool: {call['name']}")
            print(f"  Args: {call['args']}")
            print("-" * 40)

if not tool_invoked:
    print("Nessun tool è stato invocato.")
