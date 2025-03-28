# analysis/uncertainty.py

import torch
from typing import Dict
from sentence_transformers import SentenceTransformer, util

# Cache SBERT model to avoid reloading
_sbert_cache = {}

def compute_uncertainty_scores(likelihood_dict: Dict, method: str = 'lastde') -> Dict:
    """
    Compute scalar uncertainty score from token-level likelihoods.

    Args:
        likelihood_dict: Dict with keys like 'token_log_likelihoods', 'entropy_per_token', etc.
        method: which metric to use: 'lastde', 'entropy', 'logit_gap', etc.

    Returns:
        Dict with a single key: 'score', and optionally intermediates.
    """
    if method == 'entropy':
        score = likelihood_dict['entropy_per_token'].mean()
    elif method == 'lastde':
        ent = likelihood_dict['entropy_per_token']
        score = ent[-1] if ent.numel() > 0 else torch.tensor(float("nan"))
    elif method == 'logit_gap':
        logits = likelihood_dict['logits']
        if logits.dim() == 3:
            logits = logits[0]  # [seq_len, vocab] if wrapped
    elif method == "attentionsar":
        return compute_attentionsar_uncertainty(likelihoods, output)
    elif method == "bertsar":
        return compute_bert_sar_uncertainty(likelihoods, output)

        # logit gap = top1 - top2 per token
        topk = torch.topk(logits, k=2, dim=-1).values  # [seq_len, 2]
        gaps = topk[:, 0] - topk[:, 1]
        score = -gaps.mean()  # larger gap = more confident, so we invert
    else:
        raise NotImplementedError(f"Unknown uncertainty method: {method}")

    return {"score": score.item()}


def compute_attentionsar_uncertainty(likelihoods, output):
    log_likelihoods = likelihoods["token_log_likelihoods"]
    if log_likelihoods.numel() == 0:
        return {"score": float("nan")}

    attentions = output.get("log_attentions", None)
    if not attentions or not isinstance(attentions, list):
        print("⚠️ No valid attention scores for SAR. Falling back to mean log-likelihood.")
        return {"score": -log_likelihoods.mean().item()}

    # Use attention from the last layer
    last_layer_attn = attentions[-1]  # shape: [batch, heads, tgt_len, src_len]
    if isinstance(last_layer_attn, tuple):
        last_layer_attn = last_layer_attn[0]

    if last_layer_attn.ndim != 4:
        print("⚠️ Unexpected attention shape:", last_layer_attn.shape)
        return {"score": -log_likelihoods.mean().item()}

    # Average over heads and get attention on generated tokens only
    attn_weights = attn_weights / attn_weights.sum().clamp(min=1e-6)  # shape: [tgt_len, src_len]

    # Take the diagonal as a proxy for self-relevance (token attends to itself)
    diag_attn = torch.diagonal(attn_weights, dim1=0, dim2=1)  # shape: [seq_len]

    # Normalize attention weights
    attn_weights = diag_attn[:len(log_likelihoods)]
    attn_weights = attn_weights / attn_weights.sum()

    # Compute SAR as attention-weighted negative log-likelihood
    sar_score = -(log_likelihoods[:len(attn_weights)] * attn_weights).sum().item()

    return {"score": sar_score}

def compute_bert_sar_uncertainty(likelihoods, output, model_name="all-MiniLM-L6-v2"):
    log_likelihoods = likelihoods["token_log_likelihoods"]
    if log_likelihoods.numel() == 0:
        return {"score": float("nan")}

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

        return {"score": score}
    except Exception as e:
        print(f"⚠️ BERT-SAR error: {e}")
        return {"score": float("nan")}