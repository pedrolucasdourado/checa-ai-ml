"""
cluster_topics.py
────────────────────────────────────────────────────────────────────────
Agrupa os laudos de fact-checking em temas reais utilizando BERTopic e 
embeddings do sentence-transformers. 

Ideal para identificar as principais narrativas (Eleições, Saúde, etc) 
e embasar a criação de Agentes Especialistas.

Uso:
    python scripts/cluster_topics.py [--input data/processed/fact_checks_all.jsonl]
"""

import argparse
import json
from pathlib import Path
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import textwrap

def run(args):
    print(f"Carregando dados de: {args.input}")
    docs = []
    titulos = []
    
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            # Para a clusterização, título + parágrafo inicial dão o melhor contexto temático
            texto = rec.get("titulo", "") + ". " + rec.get("texto", "")[:500]
            docs.append(texto)
            titulos.append(rec.get("titulo", ""))
            
    if not docs:
        print("Nenhum documento encontrado.")
        return
        
    print(f"Total de documentos carregados: {len(docs)}")
    
    print("Gerando embeddings (paraphrase-multilingual-mpnet-base-v2)...")
    embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    embeddings = embedding_model.encode(docs, show_progress_bar=True)
    
    print("Treinando BERTopic...")
    # Ajustando min_topic_size para funcionar bem mesmo com a amostra inicial de 200 matérias
    min_size = 5 if len(docs) < 1000 else 15
    topic_model = BERTopic(
        language="multilingual",
        min_topic_size=min_size,
        calculate_probabilities=False
    )
    
    topics, _ = topic_model.fit_transform(docs, embeddings)
    
    print("\n" + "="*70)
    print("🎯 CLUSTERS DE DESINFORMAÇÃO DESCOBERTOS")
    print("="*70)
    
    freq = topic_model.get_topic_info()
    
    for idx, row in freq.iterrows():
        topic_id = row["Topic"]
        count = row["Count"]
        name = row["Name"]
        
        # O tópico -1 é o cluster de outliers (ruído) do HDBSCAN
        if topic_id == -1:
            print(f"\n[ Tópico -1 ] Outliers / Diversos ({count} matérias)")
            continue
            
        print(f"\n[ Tópico {topic_id} ] Frequência: {count} matérias")
        # Mostra as palavras-chave do tópico
        keywords = topic_model.get_topic(topic_id)
        if keywords:
            kw_str = ", ".join([kw[0] for kw in keywords[:7]])
            print(f"Palavras-chave: {kw_str}")
            
        # Pega 2 documentos representativos deste tópico
        repr_docs = topic_model.get_representative_docs(topic_id)
        if repr_docs:
            print("Exemplos de checagens:")
            for i, d in enumerate(repr_docs[:2]):
                # Acha o título correspondente para exibição
                doc_idx = docs.index(d) if d in docs else -1
                t = titulos[doc_idx] if doc_idx != -1 else d[:60]+"..."
                print(f"  - {t}")
                
    print("\n" + "="*70)
    print("Resumo estatístico salvo no console. O modelo BERTopic pode ser exportado no futuro.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/fact_checks_all.jsonl")
    args = parser.parse_args()
    run(args)
