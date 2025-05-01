import os  
import re 
import uuid 
from dotenv import load_dotenv  
from langchain_openai import ChatOpenAI 
from langchain_core.tools import tool 
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage  
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver 
from tavily import TavilyClient
#import trafilatura

# === CARICAMENTO DELLE VARIABILI D'AMBIENTE ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# === INIZIALIZZAZIONE DEI CLIENT ===
client = TavilyClient(api_key=TAVILY_API_KEY)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# === DEFINIZIONE DEI TOOLS PERSONALIZZATI ===
@tool
def web_search(topic: str) -> str:
    """Esegue una ricerca web sul topic fornito, usando Tavily."""
    if not topic:
        return "Errore: nessun topic fornito."
    try:
        results = client.search(query=topic, max_results=5)
        if not results or "results" not in results:
            return "Nessun risultato trovato."
        formatted_results = []
        for r in results["results"]:
            summary = r.get("content", "").strip()
            url = r.get("url", "")
            if url and summary:
                formatted_results.append(f"- {url}\n  → {summary[:200]}...")
        if not formatted_results:
            return "Nessun risultato utile trovato."
        return f"Risultati per '{topic}':\n" + "\n\n".join(formatted_results)
    except Exception as e:
        return f"Errore durante la ricerca: {str(e)}"
    
    #Viene fatta una ricerca web sul topic fornito, usando Tavily, e restituisce i risultati formattati. Se qualcosa va storto, restituisce un messaggio di errore 
    #sia nel caso in cui non ci siano risultati che nel caso in cui ci sia un errore durante la ricerca. Per la ricerca abbiamo usato Tavily, che richiede una chiave API.

@tool
def evaluate_source(url: str) -> str:
    """Valuta la qualità e affidabilità di una fonte fornita (URL)."""
    messages = [
        SystemMessage(content="Agisci come valutatore di fonti per un blog cinematografico."),
        HumanMessage(content=f"Sto considerando questa fonte: {url}\nValuta questa fonte e rispondi in JSON:")
    ]
    return llm.invoke(messages).content

#Questo tool si occupa di valutare la qualità e l'affidabilità di una fonte fornita (URL).
#Viene creato un messaggio di sistema che indica che il tool deve agire come valutatore di fonti per un blog cinematografico e viene passato l'URL da valutare.
#E stato pensato usando un messaggio di sistema e un messaggio umano per fornire il contesto necessario al modello.

@tool
def generate_article(prompt: str) -> str:
    """Genera un articolo cinematografico a partire da un prompt fornito."""
    messages = [
        SystemMessage(content="""Agisci come esperto cinematografico e scrivi un articolo in base al prompt dato.
                      Se non è specificato altrimenti scrivi articoli di massimo 250 parole. Devi essere creativo e usare tagli e parole diverse quando richiesto."""),
        HumanMessage(content=f"Prompt dell'articolo: {prompt}")
    ]
    return llm.invoke(messages).content

#Questo tool genera un articolo cinematografico a partire da un prompt fornito. Non da un riassunto ma un articolo completo.
#Se viene chiesto di creare un articolo su una tematica diversa, il tool non lo fa.
#Viene creato un messaggio di sistema che indica che il tool deve agire come esperto cinematografico e scrivere un articolo in base al prompt dato.
#E' stato pensato usando un messaggio di sistema e un messaggio umano per fornire il contesto necessario al modello.


@tool
def check_fact(content: str, url: str, evaluation: str) -> str:
    """Confronta il contenuto dell'articolo con una fonte e ne valuta l'accuratezza."""
    if not content or not evaluation or not url:
        return "Errore: uno o più campi richiesti sono vuoti."
    messages = [
        SystemMessage(content="Sei un fact-checker per articoli cinematografici."),
        HumanMessage(content=f"Sto scrivendo questo articolo:\n\n{content}\n\nLa fonte è: {url}\nValutazione della fonte: {evaluation}\nLa fonte conferma il contenuto?")
    ]
    return llm.invoke(messages).content

#Questo tool confronta il contenuto dell'articolo con una fonte e ne valuta l'accuratezza.
#Controlla se i campi richiesti sono vuoti e restituisce un messaggio di errore in caso affermativo.
#Viene creato un messaggio di sistema che indica che il tool deve agire come fact-checker per articoli cinematografici e viene passato il contenuto dell'articolo, l'URL della fonte e la valutazione della fonte.
#Nel caso di un checking il tool non da un errore ma fa una segnalazione


@tool
def generate_report(checked_sources: str) -> str:
    """Genera un report sull'affidabilità complessiva dell'articolo."""
    messages = [
        SystemMessage(content="Agisci come editor e ottimizzatore di contenuti per blog cinematografici."),
        HumanMessage(content="Ecco le fonti analizzate e i risultati del fact-checking:\n"
                         f"{checked_sources}\n\n"
                         "Sulla base di queste, genera un report che:\n"
                         "- Valuta l'affidabilità generale del mio articolo\n"
                         "- Indica se ci sono elementi da correggere\n"
                         "- Suggerisce miglioramenti stilistici o di contenuto")
    ]
    return llm.invoke(messages).content

@tool
def scrape_website(url: str) -> str:
    """Effettua scraping del contenuto principale di una pagina web utilizzando trafilatura."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            return "Errore: impossibile scaricare il contenuto."

        result = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        if result is None:
            return "Errore: impossibile estrarre il contenuto."

        return f"Contenuto estratto dal sito {url}:\n\n{result[:3000]}"  # Limita a 3000 caratteri
    except Exception as e:
        return f"Errore scraping: {str(e)}"
    
#Questo tool effettua scraping del contenuto principale di una pagina web utilizzando trafilatura.
#Abbiamo scelto di fare scraping per estendere le capacità del tool di ricerca web, in modo da avere un contenuto più dettagliato e preciso.

@tool
def suggest_articles() -> str:
    """Suggerisce 5 idee originali per articoli cinematografici."""
    try:
        # Messaggi per il modello
        messages = [
            SystemMessage(content="Agisci come editor creativo di un blog sul cinema."),
            HumanMessage(content="Suggeriscimi 5 idee originali per articoli brevi (<200 parole) a tema cinematografico. Scrivi solo i titoli in lista numerata. Suggerisci sempre cose diverse.")
        ]
        # Invoca il modello per generare i suggerimenti
        response = llm.invoke(messages).content
        print("\n🎥 Ecco i suggerimenti per articoli:")
        print(response)
        return response
    except Exception as e:
        return f"Errore durante la generazione dei suggerimenti: {str(e)}"
# Questo tool suggerisce 5 idee originali per articoli cinematografici.

# === BINDING DEI TOOL CON L'LLM ===
tools = [web_search, evaluate_source, generate_article, check_fact, generate_report, scrape_website, suggest_articles]
llm_with_tools = llm.bind_tools(tools=tools, tool_choice="auto")

# === NODO ASSISTENTE CHE GESTISCE LE DECISIONI ===
sys_msg = SystemMessage(content="""
Sei un assistente per un blog cinematografico.
🛑 Non puoi rispondere direttamente all'utente.  
✅ Il tuo unico compito è scegliere quale tool usare.  
🔧 Usa sempre e solo i tools disponibili, altrimenti restituisci un errore.
    Prima di generare un articolo fai SEMPRE web search per avere informazioni aggiornate.
    Dopo il web search fai Sempre scraping della fonte che ti sembra più pertinente. Poi genera l'articolo.
🚫 Se nessun tool è adatto, non rispondere.
🚫 MAI generare articoli a meno che non sia esplicitamente richiesto.
📝 Per 'articolo' si intende solo contenuti <200 parole) strutturati.
Parla solo di cinema, non di musica, religione, politica, etica o altro che non sia cinema.
""")
 
# Questo messaggio di sistema definisce il comportamento dell'assistente, specificando che non può rispondere direttamente all'utente e che deve scegliere quale tool usare.
# Tramite questo messaggio di sistema ci assicuriamo che l'assistente faccia esattamente ciò che noi desideriamo e ovviamente ne viene tenuto conto nel grafo e nei vari tools


def assistant(state: MessagesState):
    # Usa il modello per decidere quale tool invocare in base alla conversazione
   
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# === NODO PER LA REVISIONE UMANA DOPO LA GENERAZIONE DELL'ARTICOLO ===
def human_review_node(state: MessagesState):
    msgs = state["messages"]
    for i in range(len(msgs)-1, -1, -1):
        msg = msgs[i]
        if isinstance(msg, AIMessage) and msg.tool_calls:
            if any(call["name"] == "generate_article" for call in msg.tool_calls):
                article_content = msgs[i+1].content if i + 1 < len(msgs) else ""
                print("\n📝 ARTICOLO GENERATO:\n")
                print(article_content)
                feedback = input("\n🧑 Vuoi rigenerarlo con un nuovo prompt? (sì/no): ").strip().lower()
                if feedback in ["sì", "si", "y", "yes"]:
                    new_prompt = input("Inserisci il nuovo prompt per l'articolo: ").strip()
                    return {"messages": [HumanMessage(content=new_prompt)]}
                else:
                    return {"messages": msgs}
    return {"messages": msgs}
# Questo nodo si occupa della revisione umana, lo human in the loop,in particolare viene fatto un controllo per vedere se l'ultimo messaggio AI ha invocato il tool generate_article.
# se avviene questa cosa, viene generato l'articolo richiesto da prompt e con esso si da all'utente la possibilità di rigenerarlo con un nuovo prompt.
# la decisione di inserire un nuovo prompt è per dare più libertà all'utente che può decidere come farlo, dando delle specifiche e forzando il prompt. Nel caso in cui
#l'utente decide di non rigenerarlo viene ripristinato il grafo e si torna al nodo dell'assistente tramite la condizione output router.

# === NODO PER LA GESTIONE DEI SUGGERIMENTI ===
def deal_with_suggestion(state: MessagesState):
    """Gestisce i suggerimenti di articoli e chiede all'utente se vuole generare un articolo."""
    msgs = state["messages"]
    last_ai_message = next(
        (msg for msg in reversed(msgs) if isinstance(msg, AIMessage) and msg.tool_calls),
        None
    )

    if last_ai_message:
        
        # Chiedi all'utente se vuole generare un articolo
        feedback = input("\n🧑 Vuoi generare un articolo su uno di questi suggerimenti? (sì/no): ").strip().lower()
        if feedback in ["sì", "si", "y", "yes"]:
            # Chiedi quale articolo generare
            topic = input("Inserisci il titolo o il numero del suggerimento scelto: ").strip()
            prompt = f"Genera un articolo sul topic: {topic}"
            return {"messages": [HumanMessage(content=prompt)]}
        else:
            print("\n👍 Va bene, torno al nodo assistant.")
            return {"messages": msgs}

    # Se non ci sono suggerimenti validi, torna al nodo assistant
    return {"messages": msgs}
# Questo nodo gestisce i suggerimenti di articoli e chiede all'utente se vuole generare un articolo. se l'utente decide di generare un articolo, 
#verrà aggiunto un nuovo messaggio umano con il prompt per generare l'articolo. In ogni caso si torna al nodo dell'assistente.
#Ma se è stato aggiunto il messsaggio, il nodo assistant lo gestirà e lo passerà al tool generate_article.

# === ROUTER PER IL TOOL OUTPUT ===
def tool_output_router(state: MessagesState) -> str:
    # Determina se deve andare a human_review, deal_with_suggestion o tornare all’assistente
    last_ai_with_tools = next(
        (msg for msg in reversed(state["messages"]) 
         if isinstance(msg, AIMessage) and msg.tool_calls),
        None
    )
    if last_ai_with_tools:
        for call in last_ai_with_tools.tool_calls:
            if call["name"] == "generate_article":
                return "human_review"
            elif call["name"] == "suggest_articles":
                return "deal_with_suggestion"
    return "assistant"

# === 


    
# === COSTRUZIONE DEL GRAFO CONVERSAZIONALE ===
memory = MemorySaver()  # Memoria interna per salvataggio stato conversazione
builder = StateGraph(MessagesState)

# Aggiunta dei nodi
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_node("human_review", human_review_node)
builder.add_node("deal_with_suggestion", deal_with_suggestion)

# Connessioni fra nodi
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_conditional_edges("tools", tool_output_router)
builder.add_edge("human_review", "assistant")
builder.add_edge("deal_with_suggestion", "assistant")  
builder.add_edge("assistant", END)

# Compilazione del grafo
react_graph_memory = builder.compile(checkpointer=memory)

# === CONFIGURAZIONE E LOOP INTERATTIVO ===
config = {
    "configurable": {
        "thread_id": str(uuid.uuid4()), # Genera un ID unico per il thread, l'implementazione che specificava un intero come id sembrava non funzionare correttamente
        "checkpoint_ns": "article_checkpoints",
        "checkpoint_id": "article_1"
    }
}

print("\nBenvenuto! Scrivi una richiesta legata al blog di cinema. Scrivi 'no', 'basta' o 'esci' per terminare.\n")

while True:
    user_input = input("Tu: ").strip()
    if re.search(r"\b(no|basta|esci|niente|stop)\b", user_input.lower()):
        print("Va bene! Alla prossima.")
        break

    
    output = react_graph_memory.invoke({"messages": [HumanMessage(content=user_input)]}, config)
    
 # Stampa messaggi generati
    for msg in output["messages"]:
        print(f"\n🤖 {msg.type.upper()}:\n{msg.content}\n{'-'*50}")

    # Tracciamento dei tool invocati
    tool_invoked = False
    print("\n🔧 TOOL INVOCATI DURANTE IL FLUSSO:")
    for msg in output["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_invoked = True
            for call in msg.tool_calls:
                print(f"- Tool: {call['name']}")
                print(f"  Args: {call['args']}")
                print("-" * 40)

    if not tool_invoked:
        print("Nessun tool è stato invocato.")

    follow_up = input("\n🤖 Posso aiutarti con qualcos'altro? (sì / no): ").strip().lower()
    if follow_up not in ["sì", "si", "y", "yes"]:
        print("Ok! Alla prossima.")
        break