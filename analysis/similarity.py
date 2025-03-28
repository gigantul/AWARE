# analysis/similarity.py

import torch
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict


def get_sbert_similarities(sentences: List[str], model_name: str = 'all-MiniLM-L6-v2') -> torch.Tensor:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, convert_to_tensor=True)
    similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)
    return similarity_matrix.cpu().numpy()


def get_attention_weights(model_outputs: Dict, layer: int = -1, head: int = None) -> torch.Tensor:
    """
    Extract token importance from attention scores
    """
    attentions = model_outputs['attentions']  # List of layers
    selected_layer = attentions[layer]  # (batch, heads, seq_len, seq_len)
    attention_matrix = selected_layer.mean(dim=1) if head is None else selected_layer[:, head]  # (batch, seq, seq)
    token_importance = attention_matrix.mean(dim=1)  # (batch, seq_len)
    return token_importance


def get_attention_based_similarity(token_importances: torch.Tensor) -> torch.Tensor:
    if token_importances.shape[1] <= 1:
        print("⚠️ Skipping: Not enough tokens for attention-based similarity.")
        return torch.tensor([])  # or return None or np.nan

    normed = token_importances / token_importances.norm(dim=1, keepdim=True)
    similarity = torch.matmul(normed, normed.T)
    return similarity.cpu().numpy()

def compute_similarity(sentences: List[str] = None, model_outputs: Dict = None, method: str = 'sbert', **kwargs):
    """
    Compute similarity matrix from SBERT embeddings or attention maps.
    """
    if method == 'sbert':
        if sentences is None:
            raise ValueError("sentences must be provided for SBERT similarity")
        return get_sbert_similarities(sentences, **kwargs)
    elif method == 'attention':
        if model_outputs is None:
            raise ValueError("model_outputs with attentions must be provided for attention similarity")
        token_importance = get_attention_weights(model_outputs, **kwargs)
        return get_attention_based_similarity(token_importance)
    else:
        raise ValueError(f"Unknown similarity method: {method}")
