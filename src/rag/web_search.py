"""
src/rag/web_search.py
─────────────────────────────────────────────────────────────────────
Utilitário para realizar buscas na web via Tavily AI para fallback de abstenção.
"""

import os
from tavily import TavilyClient
from typing import List, Dict

def perform_web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Realiza uma busca na web usando Tavily AI e retorna os snippets dos resultados.
    """
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        print("Erro: TAVILY_API_KEY não configurada no ambiente.")
        return []

    try:
        # Inicializa o cliente da Tavily
        tavily = TavilyClient(api_key=api_key)
        
        # Realiza a busca. O parâmetro 'search_depth="advanced"' traz resultados mais profundos
        # mas para protótipos 'basic' é mais rápido e consome menos créditos.
        response = tavily.search(
            query=query, 
            search_depth="basic", 
            max_results=max_results
        )
        
        # A resposta da Tavily contém uma lista de 'results'
        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "snippet": r.get("content", ""), # Tavily usa 'content' em vez de 'body'
                "url": r.get("url", ""),
            })
            
        return results

    except Exception as e:
        print(f"Erro ao realizar busca via Tavily: {e}")
        return []
