import torch
from dataclasses import dataclass

@dataclass
class GraphRAGConfig:
    """
    Configurazione per il modello GraphRAG e il suo addestramento.
    """
    # Dimensioni degli embedding
    token_dim: int = 768
    node_dim: int = 768
    
    # Parametri di sistema
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Iperparametri di addestramento
    learning_rate: float = 1e-3
    temperature: float = 0.07
    num_negatives: int = 5
    
    # Configurazione Grafo
    use_graph_structure: bool = True
    graph_regularization_weight: float = 0.1