import os
import json
import torch
import torch.nn.functional as F
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def retrieve_simple_attention_only(
    claim, retriever, model, G, idx_to_node_id, chunk_text_map, 
    top_k_nodes=10, top_j_chunks=5, threshold=0.5, num_hops=2, decay_factor=0.5
):
    device = retriever.device
    
    # 1. Ottenere la matrice Eq (1, m, d)
    inputs = model.tokenize([claim])
    for k,v in inputs.items(): inputs[k] = v.to(device)
    
    with torch.no_grad():
        output = model[0](inputs)
        Eq = output['token_embeddings'] # (1, m, d)
        
        retriever.model.eval()
        top_indices, top_scores = retriever.retrieve(
            Eq, retriever.node_embeddings, 
            use_graph_expansion=False, 
            threshold=threshold, 
            decay_factor=decay_factor, 
            num_hops=num_hops
        )
    
    top_node_ids = [idx_to_node_id[int(idx)] for idx in top_indices]
    
    candidate_chunk_ids = set()
    
    for node_id in top_node_ids:
        if G.has_node(node_id):
            ids = G.nodes[node_id].get('text_unit_ids', [])
            candidate_chunk_ids.update(ids)
            
    subgraph = G.subgraph(top_node_ids)
    for u, v, data in subgraph.edges(data=True):
        ids = data.get('text_unit_ids', [])
        candidate_chunk_ids.update(ids)
        
    print(f"🔍 Nodi trovati: {len(top_node_ids)}. Chunk candidati unici: {len(candidate_chunk_ids)}")

    if not candidate_chunk_ids:
        return {'evidence_text': "No chunks found.", 'chunks': []}

    chunk_candidates_list = []
    chunk_texts = []
    
    for c_id in candidate_chunk_ids:
        text = chunk_text_map.get(c_id, "")
        if text and len(text.strip()) > 10: 
            chunk_candidates_list.append({'id': c_id, 'text': text})
            chunk_texts.append(text)
            
    if not chunk_texts:
         return {'evidence_text': "Chunks found but empty content.", 'chunks': []}

    with torch.no_grad():
        chunk_embeddings = model.encode(chunk_texts, convert_to_tensor=True, device=device)
        claim_embedding = model.encode([claim], convert_to_tensor=True, device=device)
        
        claim_norm = F.normalize(claim_embedding, p=2, dim=1)
        chunks_norm = F.normalize(chunk_embeddings, p=2, dim=1)
        
        sim_scores = torch.mm(claim_norm, chunks_norm.transpose(0, 1)).squeeze(0)
        
        k_actual = min(len(chunk_texts), top_j_chunks)
        best_scores, best_indices = torch.topk(sim_scores, k=k_actual)
        
    final_chunks = []
    evidence_parts = [f"=== TOP {k_actual} RELEVANT CHUNKS ==="]
    
    best_indices_list = best_indices.cpu().tolist()
    best_scores_list = best_scores.cpu().tolist()
    
    for idx, score in zip(best_indices_list, best_scores_list):
        chunk_obj = chunk_candidates_list[idx]
        
        final_chunks.append({
            'id': chunk_obj['id'],
            'score': score,
            'text': chunk_obj['text']
        })
        
        # Visualizzazione per il prompt
        evidence_parts.append(f"\nText Unit ID: {chunk_obj['id']} (Score: {score:.4f})")
        evidence_parts.append(f"Content: {chunk_obj['text']}") 
        evidence_parts.append("---")

    return {
        'evidence_text': "\n".join(evidence_parts),
        'chunks': final_chunks
    }

def run_experiment_pass_simple(
    df: pd.DataFrame,
    retriever,
    model,
    G,
    idx_to_node_id: dict,
    chunk_text_map: dict,  # AGGIUNTO
    client,
    MODEL_NAME: str,
    embedder: str,         # AGGIUNTO
    prompt_template: str,
    run_id: int,
    experiment_name: str,
    top_k: int = 10,
    top_j: int = 5,        # AGGIUNTO
    threshold: float = 0.5,
    num_hops: int = 1,
    decay_factor: float = 0.5
):
    results_data = []
    print(f"🔄 Run #{run_id} [{experiment_name}] - Elaborazione {len(df)} claims (Simple Retrieval)...")

    for index, row in df.iterrows():
        claim = row["Claim"]
        evidence_id = row.get('evidence_id', f"idx_{index}")
        ground_truth = row.get("cleaned_truthfulness", "NEI")
        
        # Filtro immagini (opzionale)
        if any(kw in str(row.get('Evidence', '')) + str(claim) for kw in ['image', 'photo']):
            continue

        try:
            retrieval_res = retrieve_simple_attention_only(
                claim=claim,
                retriever=retriever,
                model=model,
                G=G,
                idx_to_node_id=idx_to_node_id,
                top_k_nodes=top_k,          
                top_j_chunks=top_j,
                chunk_text_map=chunk_text_map,
                threshold=threshold,
                num_hops=num_hops,
                decay_factor=decay_factor
            )
            
            evidence_text = retrieval_res['evidence_text']
            print(f"Claim {index}:", claim)
            
            if not evidence_text.strip():
                evidence_text = "No relevant entity descriptions found."

            final_prompt = prompt_template.format(
                evidence=evidence_text, 
                claim=claim
            )

            if embedder == "nomic_deepseek":
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a fact-checking assistant. Output JSON only."},
                        {"role": "user", "content": final_prompt + "\nReturn a JSON with format: {\"reasoning\": \"...\", \"label\": \"SUPPORTED|REFUTED\"}"}
                    ],
                    response_format={"type": "json_object"}, 
                    temperature=0.1
                )
                raw = response.choices[0].message.content
                
                try:
                    data = json.loads(raw)
                    prediction = data.get("label", "NEI").lower()
                    full_answer = data
                except:
                    prediction = "ERROR"
                    full_answer = "ERROR"

            else:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a precise fact-checking assistant."},
                        {"role": "user", "content": final_prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.1
                )
                
                full_answer = response.choices[0].message.content.strip().lower()
                
                prediction = "NEI"
                if "final answer: supported" in full_answer: prediction = "supported"
                elif "final answer: refuted" in full_answer: prediction = "refuted"
                elif "final answer: nei" in full_answer: prediction = "NEI"
                elif "not enough information" in full_answer: prediction = "NEI"
                elif "supported" in full_answer: prediction = "supported"
                elif "refuted" in full_answer: prediction = "refuted"
                elif "nei" in full_answer: prediction = "NEI"
                elif "true" in full_answer: prediction = "supported" 
                elif "false" in full_answer: prediction = "refuted"

            print("Full Answer: ", full_answer)
            print("prediction: ", prediction)
            print("GT: ", ground_truth)

            results_data.append({
                "experiment_name": experiment_name,
                "run_id": run_id,
                "evidence_id": evidence_id,
                "original_claim": claim,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "retrieved_context": evidence_text,
                "full_llm_response": full_answer
            })
            
        except Exception as e:
            print(f"❌ Error row {index}: {e}")
            results_data.append({
                "experiment_name": experiment_name,
                "run_id": run_id,
                "evidence_id": evidence_id,
                "prediction": "ERROR",
                "ground_truth": ground_truth
            })

    return results_data

def save_summary_metrics(results_df, summary_filename="experiment_summary.csv"):
    """
    Calcola metriche aggregate (Accuracy, F1, Confusion Matrix) per ogni Run 
    di ogni Esperimento e le salva in CSV.
    """
    os.makedirs(os.path.dirname(summary_filename), exist_ok=True)
    
    print(f"\n📊 Generazione riepilogo in '{summary_filename}'...")
    
    summary_data = []
    labels = ["supported", "refuted"]
    
    if 'experiment_name' not in results_df.columns or 'run_id' not in results_df.columns:
        print("⚠️ Errore: Il dataframe non contiene le colonne 'experiment_name' o 'run_id'.")
        return pd.DataFrame()

    grouped = results_df.groupby(['experiment_name', 'run_id'])
    
    for (exp_name, run_id), group in grouped:
        y_true = group['ground_truth'].astype(str).tolist()
        y_pred = group['prediction'].astype(str).tolist()
        
        try:
            acc = accuracy_score(y_true, y_pred)
        except Exception:
            acc = 0.0

        try:
            f1 = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
        except Exception:
            f1 = 0.0
            
        try:
            cm = confusion_matrix(y_true, y_pred, labels=labels)
        except Exception:
            cm = np.zeros((3,3), dtype=int)

        row_dict = {
            "experiment_name": exp_name,
            "run_id": run_id,
            "num_samples": len(group),
            "accuracy": round(acc, 4),
            "f1_macro": round(f1, 4),
            "confusion_matrix_raw": str(cm.tolist()), 
        }
        
        for i, row_label in enumerate(labels):
            for j, col_label in enumerate(labels):
                col_name = f"Actual_{row_label}_Pred_{col_label}"
                row_dict[col_name] = cm[i, j]
                
        summary_data.append(row_dict)
    
    summary_df = pd.DataFrame(summary_data)
    
    if not summary_df.empty:
        summary_df = summary_df.sort_values(by=['experiment_name', 'run_id'])
        
    summary_df.to_csv(summary_filename, index=False)
    
    print("✅ File di riepilogo salvato con successo!")
    return summary_df