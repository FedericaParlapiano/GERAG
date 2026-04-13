import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, List, Tuple

# Importa le classi relative al tuo progetto
# (Adegua questi import in base a dove hai posizionato questi componenti)
from .config import GraphRAGConfig  # Se hai spostato la dataclass in un file config.py
from .retriever import GraphRAGRetriever # Necessario per il metodo evaluate()

class ImprovedCrossAttentionMatrix(nn.Module):
    """
    Matrice D migliorata con projection layers per gestire meglio
    la trasformazione token -> node space.
    """
    def __init__(self, config: GraphRAGConfig):
        super().__init__()
        self.config = config

        self.token_projection = nn.Sequential(
            nn.Linear(config.token_dim, config.node_dim),
            nn.LayerNorm(config.node_dim),
            nn.ReLU(),
            nn.Linear(config.node_dim, config.node_dim)
        )

        self.node_projection = nn.Sequential(
            nn.Linear(config.node_dim, config.node_dim),
            nn.LayerNorm(config.node_dim)
        )

        self.scale = np.sqrt(config.node_dim)

    def forward(self, Eq: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Eq: (batch_size, m, token_dim)
            K:  (n_nodes, node_dim)

        Returns:
            node_activation_scores: (batch_size, n_nodes)
        """

        mask = (Eq.abs().sum(dim=-1, keepdim=True) > 0)

        Q = self.token_projection(Eq)

        # (n_nodes, node_dim)
        K_proj = self.node_projection(K)

        attn_logits = torch.matmul(Q, K_proj.T) / self.scale

        attn_logits = attn_logits.masked_fill(mask == 0, -1e9)

        S_matrix = torch.softmax(attn_logits, dim=-1)

        mask = mask.float()                     # (B, m, 1)
        real_token_counts = mask.sum(dim=1)     # (B, 1)

        node_activation_scores = (
            S_matrix * mask
        ).sum(dim=1) / (real_token_counts + 1e-9)

        return node_activation_scores


class GraphRAGTrainer:
    """
    Trainer che risolve i 3 problemi principali:
    1. Loss coerente con inferenza (contrastive learning)
    2. Training diretto su (query_tokens, relevant_nodes)
    3. Integra struttura del grafo
    """
    def __init__(
        self, 
        config: GraphRAGConfig,
        node_embeddings: torch.Tensor,
        adjacency_matrix: Optional[torch.sparse.FloatTensor] = None,
        node_to_community: Optional[Dict[int, int]] = None
    ):
        """
        Args:
            config: configurazione
            node_embeddings: (num_nodes, node_dim)
            adjacency_matrix: matrice di adiacenza sparsa (opzionale)
            node_to_community: mapping nodo->community (opzionale)
        """
        self.config = config
        self.device = torch.device(config.device)
        
        self.model = ImprovedCrossAttentionMatrix(config).to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        self.node_embeddings = node_embeddings.to(self.device)
        self.adjacency_matrix = adjacency_matrix
        self.node_to_community = node_to_community
        
        self.node_embeddings = F.normalize(self.node_embeddings, dim=1)
        
        print(f"🚀 Improved Model initialized on {self.device}")
        print(f"   Nodes: {len(node_embeddings)}, Dim: {config.node_dim}")
        
    def compute_contrastive_loss(
        self, 
        query_tokens_matrix: torch.Tensor, # (batch_size, m, token_dim)
        positive_node_ids: List[int],
        hard_negatives: Optional[List[List[int]]] = None
    ) -> Tuple[torch.Tensor, float]:
        """
        Contrastive loss basata sugli score di attivazione S(vi).
        """
        batch_size = len(positive_node_ids)
        
        all_node_scores = self.model(query_tokens_matrix, self.node_embeddings)
        
        positive_logits = all_node_scores[range(batch_size), positive_node_ids].unsqueeze(1) # (batch, 1)
        
        if hard_negatives is None:
            negative_ids = torch.randint(
                0, len(self.node_embeddings), (batch_size, self.config.num_negatives)
            ).to(self.device)
        else:
            negative_ids = torch.tensor(hard_negatives).to(self.device)
            
        negative_logits = torch.gather(all_node_scores, 1, negative_ids)
        
        logits = torch.cat([positive_logits, negative_logits], dim=1) / self.config.temperature
        
        labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        loss = F.cross_entropy(logits, labels)
        
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == labels).float().mean().item()
    
        return loss, accuracy
    
    def compute_graph_regularization(
        self, 
        query_tokens_matrix: torch.Tensor,
        positive_node_ids: List[int]
    ) -> torch.Tensor:
        """
        Regolarizzazione: l'attivazione sui nodi deve essere smooth
        rispetto alla struttura del grafo.
        """
        batch_size = len(positive_node_ids)
    
        if self.adjacency_matrix is None:
            return torch.tensor(0.0, device=self.device)
            
        node_scores = self.model(
            query_tokens_matrix, 
            self.node_embeddings
        )
    
        reg_loss = 0.0
    
        for i, node_id in enumerate(positive_node_ids):
            neighbors = self.adjacency_matrix[node_id].coalesce().indices()[0]
    
            if len(neighbors) > 0:
                pos_score = node_scores[i, node_id]
    
                neighbor_scores = node_scores[i, neighbors]
    
                reg_loss += torch.mean((neighbor_scores - pos_score) ** 2)
    
        return reg_loss / batch_size

    def train_step(
        self, 
        batch_queries: torch.Tensor,
        batch_positive_ids: List[int],
        batch_hard_negatives: Optional[List[List[int]]] = None
    ) -> Dict[str, float]:
        """
        Singolo step di training.
        
        Args:
            batch_queries: (batch_size, num_tokens, token_dim)
            batch_positive_ids: lista di node_id rilevanti
            batch_hard_negatives: opzionale, hard negatives pre-minati
        """
        self.optimizer.zero_grad()
        
        loss, acc = self.compute_contrastive_loss(
            batch_queries, 
            batch_positive_ids,
            batch_hard_negatives
        )
        
        if self.config.use_graph_structure:
            reg_loss = self.compute_graph_regularization(
                batch_queries, 
                batch_positive_ids
            )
            total_loss = self.config.graph_regularization_weight*loss + (1-self.config.graph_regularization_weight) * reg_loss  # peso della regolarizzazione
        else:
            total_loss = loss
            reg_loss = 0.0
        
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "reg_loss": reg_loss if isinstance(reg_loss, float) else reg_loss.item(),
            "accuracy": acc
        }
    
    def train(
        self, 
        train_data: List[Tuple[torch.Tensor, int]],
        num_epochs: int,
        batch_size: int = 32,
        val_data: Optional[List[Tuple[torch.Tensor, int]]] = None
    ):
        """
        Training loop completo.
        
        Args:
            train_data: lista di tuple (query_tokens, relevant_node_id)
            num_epochs: numero di epoche
            batch_size: dimensione batch
            val_data: dati di validazione (opzionale)
        """
        num_samples = len(train_data)
        
        for epoch in range(num_epochs):
            self.model.train()
            
            indices = np.random.permutation(num_samples)
            
            epoch_metrics = {"loss": [], "reg_loss": [], "accuracy": []}
            
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                
                batch_queries = []
                batch_positives = []
                
                for idx in batch_indices:
                    query_tokens, positive_id = train_data[idx]
                    batch_queries.append(query_tokens)
                    batch_positives.append(positive_id)
                
                batch_queries = torch.stack(batch_queries).to(self.device)
                
                metrics = self.train_step(batch_queries, batch_positives)
                
                for key, value in metrics.items():
                    epoch_metrics[key].append(value)
            
            avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
            
            val_metrics = {}
            if val_data and (epoch + 1) % 5 == 0:
                pass
            
            if (epoch + 1) % 1 == 0 or val_metrics:
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                print(f"  Train Loss: {avg_metrics['loss']:.4f}, "
                      f"Acc: {avg_metrics['accuracy']:.3f}")
                if val_metrics:
                    print(f"  Val Hits@5: {val_metrics['hits@5']:.3f}, "
                          f"MRR: {val_metrics['mrr']:.3f}")
    
    @torch.no_grad()
    def evaluate(
        self, 
        val_data: List[Tuple[torch.Tensor, int]], 
        k: int = 5
    ) -> Dict[str, float]:
        """
        Valuta il modello su validation set.
        """
        self.model.eval()

        from .retriever import GraphRAGRetriever
        
        retriever = GraphRAGRetriever(
            self.model, 
            self.node_embeddings,
            adjacency_matrix=self.adjacency_matrix
        )
        
        hits_at_k = []
        mrr_scores = []
        
        for query_tokens, relevant_node in val_data:
            query_tokens = query_tokens.to(self.device)
            
            # Retrieve top-k
            retrieved_ids, _ = retriever.retrieve(query_tokens, top_k=k)
            
            # Hits@K
            hits = relevant_node in retrieved_ids
            hits_at_k.append(hits)
            
            # MRR
            if relevant_node in retrieved_ids:
                rank = retrieved_ids.index(relevant_node) + 1
                mrr_scores.append(1.0 / rank)
            else:
                mrr_scores.append(0.0)
        
        return {
            "hits@5": np.mean(hits_at_k),
            "mrr": np.mean(mrr_scores)
        }
