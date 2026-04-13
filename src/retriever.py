import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

class GraphRAGRetriever:
    """
    Retriever allineato alla teoria: usa i punteggi di attivazione S(vi)
    calcolati via Cross-Attention sui token.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        node_embeddings: torch.Tensor,
        adjacency_matrix: Optional[torch.sparse.FloatTensor] = None,
        device: str = "cuda"
    ):
        self.model = model
        self.model.eval()
        self.device = device
        
        self.node_embeddings = node_embeddings.to(self.device).detach()
        
        if adjacency_matrix is not None:
            self.adjacency_matrix = adjacency_matrix.to(self.device)
        else:
            self.adjacency_matrix = None
        
    @torch.no_grad()
    def retrieve(
        self, 
        query_tokens: torch.Tensor, # Input: (m, d) oppure (1, m, d)
        top_k: int = 10,
        use_graph_expansion: bool = False,
        num_hops: int = 1,
        decay_factor: float = 0.1,
        threshold: float = 0.5
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Calcola i punteggi S(vi) per ogni nodo e recupera i migliori.
        """
        query_tokens = query_tokens.to(self.device)
        
        if query_tokens.dim() == 2:
            query_tokens = query_tokens.unsqueeze(0)
            
        all_scores = self.model(query_tokens, self.node_embeddings)
        
        all_scores = all_scores.squeeze(0)
        
        if use_graph_expansion and self.adjacency_matrix is not None:
            current_scores = all_scores
            expanded_scores = all_scores.clone()
            
            for hop in range(num_hops):
                neighbor_scores = torch.sparse.mm(
                    self.adjacency_matrix,
                    current_scores.unsqueeze(1)
                ).squeeze(1)
                
                decay = decay_factor ** (hop + 1)
                expanded_scores += decay * neighbor_scores
                current_scores = neighbor_scores
            
            all_scores = expanded_scores
            
            temperature = 0.5
            all_scores = all_scores/sum(all_scores)
            
        sorted_scores, sorted_indices = torch.sort(all_scores, descending=True)
        
        cumulative_mass = torch.cumsum(sorted_scores, dim=0)
                
        k_dynamic = torch.searchsorted(cumulative_mass, threshold).item() + 1
        
        top_indices = sorted_indices[:k_dynamic]
        top_scores = sorted_scores[:k_dynamic]  
        
        return top_indices, top_scores