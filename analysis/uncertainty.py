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

    logits = logits.to(embedding_matrix.device)  # 🔧 Brute force fix
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


def compute_aware_uncertainty(likelihoods, output):
    logits = likelihoods.get("logits")
    attentions = output.get("log_attentions")
    embedding_matrix = output.get("embedding_matrix")

    # 🔍 Debug Print
    print("📦 AWARE Debug — Input Check:")
    print(f" - logits: {'✅' if logits is not None else '❌'}")
    print(f" - logits shape: {getattr(logits, 'shape', 'N/A')}")
    print(f" - attentions: {'✅' if attentions is not None else '❌'}")
    print(f" - embedding_matrix: {'✅' if embedding_matrix is not None else '❌'}")
    print(f" - embedding shape: {getattr(embedding_matrix, 'shape', 'N/A')}")
    print("=" * 50)

    if logits is None or attentions is None or embedding_matrix is None:
        print("⚠️ Missing logits, attention, or embeddings for AWARE.")
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

# compute_attentionsar_uncertainty, compute_bert_sar_uncertainty remain unchanged
# Helper functions for attentionsar and bertsar
def compute_attentionsar_uncertainty(likelihoods, output):
    log_likelihoods = likelihoods["token_log_likelihoods"]
    if log_likelihoods.numel() == 0:
        return float("nan")

    attentions = output.get("log_attentions", None)
    if not attentions or not isinstance(attentions, list):
        print("⚠️ No valid attention scores for SAR. Falling back to mean log-likelihood.")
        return -log_likelihoods.mean().item()

    # Use attention from the last layer: [batch, heads, tgt_len, src_len]
    last_layer_attn = attentions[-1]
    if isinstance(last_layer_attn, tuple):
        last_layer_attn = last_layer_attn[0]

    if last_layer_attn.ndim != 4:
        print("⚠️ Unexpected attention shape:", last_layer_attn.shape)
        return -log_likelihoods.mean().item()

    # Mean over heads → [tgt_len, src_len]
    attn_weights = last_layer_attn.mean(dim=1).squeeze(0)  # Remove batch dim

    # For each token t, compute attention-weighted uncertainty from tokens 0 to t-1
    weighted_ll = []
    for t in range(1, len(log_likelihoods)):
        past_attn = attn_weights[t, :t]  # attention from token t to previous tokens
        past_attn = past_attn / past_attn.sum()  # normalize
        past_ll = log_likelihoods[:t]
        weighted_ll.append((past_ll * past_attn).sum())

    # Optional: add token 0's unweighted log-likelihood (no context)
    weighted_ll = [log_likelihoods[0]] + weighted_ll

    sar_score = -torch.stack(weighted_ll).mean().item()
    return sar_score

def compute_bert_sar_uncertainty(likelihoods, output, model_name="all-MiniLM-L6-v2"):
    log_likelihoods = likelihoods["token_log_likelihoods"]
    if log_likelihoods.numel() == 0:
        return float("nan")

    if model_name not in _sbert_cache:
        _sbert_cache[model_name] = SentenceTransformer(model_name)
    sbert = _sbert_cache[model_name]

    try:
        # Get the tokens (optional fallback)
        input_text = output.get("input_text", "")
        answer_text = output.get("generated_text", "")

        # Tokenize into individual words (rough approximation)
        tokens = input_text.split()
        token_embeddings = sbert.encode(tokens, convert_to_tensor=True, normalize_embeddings=True)
        answer_embedding = sbert.encode(answer_text, convert_to_tensor=True, normalize_embeddings=True)

        # Compute cosine similarity between each token and the answer
        similarities = util.pytorch_cos_sim(token_embeddings, answer_embedding).squeeze()

        # Normalize
        similarities = similarities[:len(log_likelihoods)]
        weights = similarities / similarities.sum().clamp(min=1e-6)

        # Weighted negative log-likelihood
        score = -(log_likelihoods[:len(weights)] * weights).sum().item()

        return score
    except Exception as e:
        print(f"⚠️ BERT-SAR error: {e}")
        return float("nan")