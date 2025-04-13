import os
import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
from dotenv import load_dotenv

# === SETUP ===
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)

# === AGENT NODE LOGIC ===

def extract_topic(prompt):
    return prompt.strip()

def input_parser_node(inputs):
    topic = extract_topic(inputs["prompt"])
    return {
        "topic": topic,
        "prompt": inputs["prompt"],
        "content": ""  # Al momento non c'è un contenuto fisso, ma sarà generato dinamicamente
    }

def search_web(topic):
    url = f"https://html.duckduckgo.com/html/?q={topic}+site:gazzetta.it+OR+site:espn.com"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    
    results = []
    for link in soup.select(".result__a")[:5]:
        href = link.get("href")
        if href:
            results.append(href)
    return results

def web_search_node(inputs):
    results = search_web(inputs["topic"])
    return {"results": results, **inputs}

def evaluate_sources(urls):
    evaluations = []
    for url in urls:
        messages = [
            SystemMessage(content="Agisci come valutatore di fonti per un blog sportivo."),
            HumanMessage(content=(
                f"Sono un blogger e sto scrivendo un articolo sportivo.\n"
                f"Sto considerando questa fonte: {url}\n"
                f"Valuta questa fonte e rispondi in JSON:\n"
                '{"score": int (1-10), "comment": "breve motivazione"}'
            ))
        ]
        response = llm.invoke(messages).content
        evaluations.append({"url": url, "evaluation": response})
    return evaluations

def source_evaluator_node(inputs):
    evaluated = evaluate_sources(inputs["results"])
    return {"evaluated": evaluated, **inputs}

def check_facts(evaluated_sources, content):
    results = []
    for item in evaluated_sources:
        url = item["url"]
        evaluation = item["evaluation"]
        messages = [
            SystemMessage(content="Sei un fact-checker per articoli sportivi."),
            HumanMessage(content=(
                f"Sto scrivendo questo articolo:\n\n{content}\n\n"
                f"La fonte è: {url}\n"
                f"Valutazione della fonte: {evaluation}\n"
                "La fonte conferma il contenuto? Rispondi spiegando eventuali discrepanze o conferme."
            ))
        ]
        response = llm.invoke(messages).content
        results.append({"url": url, "fact_check": response})
    return results

def fact_checker_node(inputs):
    checked = check_facts(inputs["evaluated"], inputs["content"])
    return {"checked": checked, **inputs}

def generate_report(checked_sources):
    messages = [
        SystemMessage(content="Agisci come editor e ottimizzatore di contenuti per blog sportivi."),
        HumanMessage(content=(
            "Ecco le fonti analizzate e i risultati del fact-checking:\n"
            f"{checked_sources}\n\n"
            "Sulla base di queste, genera un report che:\n"
            "- Valuta l'affidabilità generale del mio articolo\n"
            "- Indica se ci sono elementi da correggere\n"
            "- Suggerisce miglioramenti stilistici o di contenuto"
        ))
    ]
    report = llm.invoke(messages).content
    return report

def report_generator_node(inputs):
    report = generate_report(inputs["checked"])
    return {"report": report}

# === GENERAZIONE DELL'ARTICOLO (NUOVO BLOCCO) ===

def generate_article(prompt):
    messages = [
        SystemMessage(content="Agisci come esperto sportivo e scrivi un articolo in base al prompt dato."),
        HumanMessage(content=f"Prompt dell'articolo: {prompt}")
    ]
    response = llm.invoke(messages).content
    return response.strip()

# === GRAPH SETUP ===

class BlogAgentState(TypedDict):
    prompt: str
    content: str
    topic: str
    results: List[str]
    evaluated: List[dict]
    checked: List[dict]
    report: Optional[str]

builder = StateGraph(BlogAgentState)

builder.add_node("InputParser", input_parser_node)
builder.add_node("GenerateArticle", generate_article)  # Aggiunto il nodo di generazione dell'articolo
builder.add_node("WebSearch", web_search_node)
builder.add_node("SourceEvaluator", source_evaluator_node)
builder.add_node("FactChecker", fact_checker_node)
builder.add_node("ReportGenerator", report_generator_node)

builder.set_entry_point("InputParser")
builder.add_edge("InputParser", "GenerateArticle")
builder.add_edge("GenerateArticle", "WebSearch")
builder.add_edge("WebSearch", "SourceEvaluator")
builder.add_edge("SourceEvaluator", "FactChecker")
builder.add_edge("FactChecker", "ReportGenerator")
builder.add_edge("ReportGenerator", END)

graph = builder.compile()

# === TEST ===

prompt = "fammi un articolo su tadej pogachar."


result = graph.invoke({
    "prompt": prompt,
    "content": ""
})

result["report"] 