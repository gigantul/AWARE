# Updated uncertainty.py with AWARE support (PCA-based epistemic entropy, edge-case safe)
import torch
from typing import Dict
from sentence_transformers import SentenceTransformer, util
import torch.nn.functional as F

# Cache SBERT model to avoid reloading
_sbert_cache = {}

# --- NEW: Lexicon collapsing using PCA-style semantic axes ---
def collapse_logits_by_semantic_axes(logits, embedding_matrix, top_k=200, variance_threshold=0.95):
    """
    Collapse logits across semantically meaningful axes using SVD-based dimensionality reduction.

    Args:
        logits: [vocab_size] tensor of logit probabilities
        embedding_matrix: [vocab_size, hidden_size] embedding vectors (tied input/output embeddings)
        top_k: how many top tokens to consider from the softmax distribution
        variance_threshold: float, amount of variance to preserve (e.g., 0.95)

    Returns:
        Tensor of reduced, semantically collapsed probabilities
    """
    probs = F.softmax(logits, dim=-1)  # [vocab_size]
    top_values, top_indices = torch.topk(probs, k=min(top_k, logits.size(0)))

    # Handle cases where there's too little to collapse meaningfully
    if top_values.numel() < 3:
        return probs[top_indices] / probs[top_indices].sum().clamp(min=1e-9)

    top_embeddings = embedding_matrix[top_indices]  # [top_k, hidden_size]

    # Center embeddings
    X = top_embeddings - top_embeddings.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)

    if S.sum().item() < 1e-6:
        return probs[top_indices] / probs[top_indices].sum().clamp(min=1e-9)

    # Compute cumulative explained variance
    explained_variance = (S**2) / (S**2).sum()
    cumulative_variance = explained_variance.cumsum(0)
    k_components = (cumulative_variance < variance_threshold).sum().item() + 1

    reduced = X @ Vh[:k_components].T  # [top_k, k_components]
    rounded = torch.round(reduced * 3)
    unique, inverse_indices = torch.unique(rounded, dim=0, return_inverse=True)

    if len(unique) <= 1:
        return probs[top_indices] / probs[top_indices].sum().clamp(min=1e-9)

    collapsed = []
    for i in range(len(unique)):
        group_mask = (inverse_indices == i)
        collapsed.append(top_values[group_mask].sum())

    collapsed_probs_tensor = torch.stack(collapsed)
    collapsed_probs_tensor = collapsed_probs_tensor / collapsed_probs_tensor.sum().clamp(min=1e-9)
    return collapsed_probs_tensor

# --- NEW: ESE computation ---
def compute_aware_uncertainty(likelihoods, output):
    logits = likelihoods.get("logits")
    attentions = output.get("log_attentions")
    embedding_matrix = output.get("embedding_matrix")

    if logits is None or attentions is None or embedding_matrix is None:
        print("⚠️ Missing logits, attention, or embeddings for ESE.")
        return float("nan")

    if isinstance(logits, tuple): logits = logits[0]
    if logits.dim() == 3: logits = logits[0]

    last_attn = attentions[-1]
    if isinstance(last_attn, tuple): last_attn = last_attn[0]
    if last_attn.ndim != 4:
        print("⚠️ Unexpected attention shape.")
        return float("nan")

    attn_weights = last_attn.mean(dim=1).squeeze(0)  # [tgt, src]
    probs = F.softmax(logits, dim=-1)
    aware_scores = []

    for t in range(probs.shape[0]):
        # Handle t=0 (no previous context): uniform self-weight
        attn_vec = attn_weights[t, :t+1] if t > 0 else torch.ones(1)
        attn_vec = attn_vec / attn_vec.sum().clamp(min=1e-6)
        weighted_logit = probs[:t+1].transpose(0, 1) @ attn_vec  # [vocab_size]

        collapsed = collapse_logits_by_semantic_axes(weighted_logit, embedding_matrix)
        entropy = -(collapsed * torch.log(collapsed.clamp(min=1e-9))).sum()
        aware_scores.append(entropy)

    if not aware_scores:
        return float("nan")

    return torch.stack(aware_scores).mean().item()


def compute_uncertainty_scores(likelihood_dict: Dict, output: Dict, methods: list = ['entropy', 'lastde', 'lastn_entropy', 'logit_gap', 'attentionsar', 'bertsar']) -> Dict:
    scores = {}

    for method in methods:
        try:
            if method == 'entropy':
                scores['entropy'] = likelihood_dict['entropy_per_token'].mean().item()

            elif method == 'lastde':
                ent = likelihood_dict['entropy_per_token']
                scores['lastde'] = ent[-1].item() if ent.numel() > 0 else float("nan")

            elif method == 'lastn_entropy':
                n = 10
                ent = likelihood_dict['entropy_per_token']
                scores['lastn_entropy'] = ent[-n:].mean().item() if ent.numel() >= n else ent.mean().item()

            elif method == 'logit_gap':
                logits = likelihood_dict['logits']
                if isinstance(logits, tuple): logits = logits[0]
                if logits.dim() == 3: logits = logits[0]
                if logits.dim() != 2:
                    print(f"⚠️ Unexpected logits shape: {logits.shape}")
                    scores['logit_gap'] = float("nan")
                    continue
                topk = torch.topk(logits, k=2, dim=-1).values
                gaps = topk[:, 0] - topk[:, 1]
                scores['logit_gap'] = -gaps.mean().item()

            elif method == 'attentionsar':
                scores['attentionsar'] = compute_attentionsar_uncertainty(likelihood_dict, output)

            elif method == 'bertsar':
                scores['bertsar'] = compute_bert_sar_uncertainty(likelihood_dict, output)

            elif method == 'aware':
                scores['aware'] = compute_aware_uncertainty(likelihood_dict, output)

            else:
                print(f"⚠️ Unknown uncertainty method: {method}")
                scores[method] = float("nan")

        except IndexError as e:
            print(f"⚠️ Skipping IndexError: {e}")
            scores[method] = float("nan")

    return scores

# Reuse from original...
# compute_attentionsar_uncertainty, compute_bert_sar_uncertainty remain unchanged
