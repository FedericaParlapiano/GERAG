import os
import ast
import pandas as pd
import numpy as np
import networkx as nx
import torch
from typing import Dict, List, Tuple, Any, Set, Optional

def load_graphrag_artifacts(artifacts_path: str, community_level: int = None) -> Tuple[nx.DiGraph, List[Set[str]], Dict[str, str]]:
    """
    Legge il grafo, le entità e le relazioni generati da Microsoft GraphRAG.
    Restituisce il grafo NetworkX, le community e il mapping dei chunk di testo.
    """
    print(f"Loading data from: {artifacts_path}")
    
    try:
        df_entities = pd.read_parquet(os.path.join(artifacts_path, "entities.parquet"))
        df_rels = pd.read_parquet(os.path.join(artifacts_path, "relationships.parquet"))
        df_communities = pd.read_parquet(os.path.join(artifacts_path, "communities.parquet"))
        df_text_units = pd.read_parquet(os.path.join(artifacts_path, "text_units.parquet"))
    except Exception as e:
        print(f"Error loading parquets: {e}")
        return nx.DiGraph(), [], {}

    chunk_text_map = dict(zip(df_text_units['id'], df_text_units['text']))
    print(f"Indexed {len(chunk_text_map)} chunks (Text Units).")

    def parse_ids(id_field: Any) -> List[str]:
        if id_field is None: return []
        if isinstance(id_field, (list, np.ndarray)): return list(id_field)
        if isinstance(id_field, str):
            try: return ast.literal_eval(id_field)
            except: return [] # Fallback
        return []

    id_to_title = dict(zip(df_entities['id'], df_entities['title']))
    valid_titles = set(df_entities['title'])
    
    G = nx.DiGraph()
    
    # Aggiunta Nodi
    for _, row in df_entities.iterrows():
        chunk_ids = parse_ids(row.get('text_unit_ids'))
        G.add_node(
            row['title'], 
            description=row.get('description', ''), 
            text_unit_ids=chunk_ids
        )
        
    # Aggiunta Archi
    for _, row in df_rels.iterrows():
        src = row['source']
        tgt = row['target']
        if src in id_to_title: src = id_to_title[src]
        if tgt in id_to_title: tgt = id_to_title[tgt]
        
        if src in valid_titles and tgt in valid_titles:
            chunk_ids = parse_ids(row.get('text_unit_ids'))
            G.add_edge(
                src, tgt, 
                description=row.get('description', ''),
                text_unit_ids=chunk_ids
            )

    # Costruzione Communities
    communities_list = []
    if community_level is not None:
        df_communities = df_communities[df_communities['level'] == community_level]
        
    for idx, row in df_communities.iterrows():
        raw_ids = row['entity_ids']
        entity_ids_list = []
        if raw_ids is None: continue
        elif isinstance(raw_ids, (list, np.ndarray)): entity_ids_list = raw_ids
        elif isinstance(raw_ids, str):
            try: entity_ids_list = ast.literal_eval(raw_ids)
            except: entity_ids_list = raw_ids.replace('[','').replace(']','').replace("'", "").split(',')

        current_community_set = set()
        for eid in entity_ids_list:
            eid_str = str(eid).strip()
            if eid_str in valid_titles: 
                current_community_set.add(eid_str)
            elif eid_str in id_to_title: 
                current_community_set.add(id_to_title[eid_str])
        
        if len(current_community_set) > 0:
            communities_list.append(current_community_set)
            
    # Fallback se non ci sono communities
    if len(communities_list) == 0:
        communities_list = list(nx.community.louvain_communities(G))

    return G, communities_list, chunk_text_map

def networkx_to_torch_sparse(G: nx.Graph, node_id_to_idx: Dict[str, int], device: str = "cuda") -> Optional[torch.sparse.FloatTensor]:
    """
    Converte un grafo NetworkX in una matrice di adiacenza sparsa di PyTorch.
    """
    edges = []
    for source, target in G.edges():
        if source not in node_id_to_idx or target not in node_id_to_idx:
            continue
            
        source_idx = node_id_to_idx[source]
        target_idx = node_id_to_idx[target]
        edges.append([source_idx, target_idx])
        if not G.is_directed():
            edges.append([target_idx, source_idx])
    
    if not edges:
        return None
    
    edges_tensor = torch.tensor(edges, dtype=torch.long).T
    num_nodes = len(node_id_to_idx)
    values = torch.ones(edges_tensor.shape[1], dtype=torch.float32)
    
    adj_matrix = torch.sparse_coo_tensor(
        edges_tensor,
        values,
        (num_nodes, num_nodes)
    )
    
    return adj_matrix.to(device)

def prepare_training_data_from_memory(
    G: nx.Graph, 
    node_id_to_idx: Dict[str, int], 
    model, 
    device: str, 
    max_length: int = 128
) -> List[Tuple[torch.Tensor, int]]:
    """
    Genera il dataset di training creando token embeddings dalle descrizioni degli archi.
    """
    print(f"🛠️ Generazione dati di training (Token Sequences fisse a {max_length})...")
    
    queries = []
    target_indices = []
    
    for u, v, data in G.edges(data=True):
        desc = data.get('description', '')
        if not desc or len(desc) < 5:
            continue
            
        if u in node_id_to_idx and v in node_id_to_idx:
            queries.append(desc)
            target_indices.append(node_id_to_idx[u])
            
            queries.append(desc)
            target_indices.append(node_id_to_idx[v])
        
    print(f"🔤 Elaborazione di {len(queries)} descrizioni...")
    
    training_samples = []
    batch_size = 32
    model.to(device)
    
    if hasattr(model, 'max_seq_length'):
        original_max_len = model.max_seq_length
        model.max_seq_length = max_length
    
    with torch.no_grad():
        for i in range(0, len(queries), batch_size):
            batch_texts = queries[i:i+batch_size]
            
            encoded = model.tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,      
                max_length=max_length,
                return_tensors="pt"
            )
            
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            model_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            output = model[0](model_inputs)
            
            Eq_batch = output['token_embeddings'] # (Batch, max_length, dim)
            
            mask_expanded = attention_mask.unsqueeze(-1).float()
            Eq_batch_masked = Eq_batch * mask_expanded
            
            for j, Eq in enumerate(Eq_batch_masked):
                training_samples.append((Eq.cpu(), target_indices[i+j]))
    
    if hasattr(model, 'max_seq_length'):
        model.max_seq_length = original_max_len
        
    return training_samples