# Updated uncertainty.py with AWARE support (PCA-based epistemic entropy, edge-case safe)
import torch
from typing import Dict
from sentence_transformers import SentenceTransformer, util
import torch.nn.functional as F

# Cache SBERT model to avoid reloading
_sbert_cache = {}

def collapse_logits_toward_question(
    logits: torch.Tensor,
    embedding_matrix: torch.Tensor,
    question_embedding: torch.Tensor,
    top_k: int = 50,
    subspace_rank: int = 6  # Number of orthogonal directions to span the answer space
) -> torch.Tensor:
    """
    Collapses the model’s output distribution into a semantic subspace
    aligned with the question and its likely answer space.

    Uses a fixed geometric threshold (epsilon) to retain only those tokens
    whose embeddings lie meaningfully within the constructed subspace.

    Args:
        logits: Unnormalized logits [vocab_size]
        embedding_matrix: Token embedding matrix [vocab_size, dim]
        question_embedding: Query vector [dim]
        top_k: Number of top-probability tokens to consider
        subspace_rank: Dimensionality of the semantic answer subspace

    Returns:
        Collapsed distribution over selected tokens (normalized)
    """

    logits = logits.to(embedding_matrix.device)
    probs = F.softmax(logits, dim=-1)

    # Select top-K tokens with highest probability
    top_values, top_indices = torch.topk(probs, k=min(top_k, logits.size(0)))
    top_embeddings = embedding_matrix[top_indices]  # [top_k, dim]

    # Construct answer subspace from question + top token embeddings
    seed_vectors = [question_embedding] + [top_embeddings[i] for i in range(min(subspace_rank - 1, top_k))]
    seed_matrix = torch.stack(seed_vectors)  # [r, dim]

    # Orthonormalize to define semantic subspace
    Q, _ = torch.linalg.qr(seed_matrix.T)  # [dim, r]
    projection_basis = Q  # [dim, r]

    # Project all top-k embeddings onto subspace and compute residuals
    projections = (top_embeddings @ projection_basis) @ projection_basis.T
    residuals = (top_embeddings - projections).norm(dim=-1)

    # 🔒 Fixed geometric radius threshold
    epsilon = 0.8
    strong_mask = residuals < epsilon
    if strong_mask.sum() < 1:
        # fallback: retain closest token
        strong_mask[torch.argmin(residuals)] = True

    # Collapse selected tokens into a normalized distribution
    selected_values = top_values[strong_mask]
    collapsed = selected_values / selected_values.sum().clamp(min=1e-9)

    return collapsed

# == INSERTED IAC-BASED TOKEN FILTERING ==
def compute_iac(attentions: list) -> torch.Tensor:
    """
    Compute Incoming Attention Centrality (IAC) for each token based on attention rollout.
    Args:
        attentions: List of attention tensors per layer (each [batch, heads, tgt, src])
    Returns:
        iac: Tensor of shape [seq_len] with normalized IAC scores
    """
    attention_rollout = None
    for layer_attn in attentions:
        if isinstance(layer_attn, tuple):
            layer_attn = layer_attn[0]  # discard cross-attn if present
        layer_attn = layer_attn.mean(dim=1).squeeze(0)  # [tgt, src]
        attention_rollout = layer_attn if attention_rollout is None else layer_attn @ attention_rollout

    incoming_attention = attention_rollout.sum(dim=0)  # [src_len]
    iac = incoming_attention / incoming_attention.sum().clamp(min=1e-6)
    return iac



# == MODIFIED compute_aware_uncertainty TO APPLY IAC FILTERING ==
def compute_aware_uncertainty(likelihoods: Dict, output: Dict, question_embedding: torch.Tensor) -> float:
    logits = likelihoods.get("logits")
    attentions = output.get("log_attentions")
    embedding_matrix = output.get("embedding_matrix")

    if logits is None or attentions is None or embedding_matrix is None:
        print("\u26a0\ufe0f Missing logits, attention, or embeddings for AWARE.")
        return float("nan")

    logits = logits.float()
    embedding_matrix = embedding_matrix.float()

    # Compute IAC and prune tokens with low epistemic salience
    iac = compute_iac(attentions)  # shape [seq_len]
    threshold = 0.05  # absolute IAC threshold (can be tuned later)
    valid_indices = torch.nonzero(iac > threshold).squeeze(-1)

    if valid_indices.numel() == 0:
        print("\u26a0\ufe0f No tokens passed IAC threshold.")
        return float("nan")

    # Softmax on logits (still in vocab space)
    probs = torch.softmax(logits, dim=-1).to(logits.device)

    aware_scores = []
    for t in valid_indices:
        t = t.item()
        attn_vec = compute_iac(attentions)[:t+1] if t > 0 else torch.ones(1, device=logits.device)
        attn_vec = attn_vec / attn_vec.sum().clamp(min=1e-6)

        weighted_logit = probs[:t+1].transpose(0, 1) @ attn_vec  # [vocab_size]
        weighted_logit = weighted_logit.to(embedding_matrix.device)

        collapsed = collapse_logits_toward_question(weighted_logit, embedding_matrix, question_embedding)
        entropy = -(collapsed * torch.log(collapsed.clamp(min=1e-9))).sum()
        aware_scores.append(entropy)

    return torch.stack(aware_scores).mean().item() if aware_scores else float("nan")

def compute_uncertainty_scores(
    likelihood_dict: Dict,
    output: Dict,
    methods: list = ['entropy', 'lastde', 'lastn_entropy', 'logit_gap', 'attentionsar', 'bertsar', 'aware']
) -> Dict:
    scores = {}

    for method in methods:
        try:
            if method in ['entropy', 'lastde', 'lastn_entropy']:
                ent = likelihood_dict.get('entropy_per_token', [])
                if isinstance(ent, list):
                    ent = torch.tensor(ent)
                if ent.numel() == 0:
                    scores[method] = float("nan")
                    continue

                if method == 'entropy':
                    scores['entropy'] = ent.mean().item()

                elif method == 'lastde':
                    scores['lastde'] = ent[-1].item()

                elif method == 'lastn_entropy':
                    n = 10
                    scores['lastn_entropy'] = ent[-n:].mean().item() if ent.numel() >= n else ent.mean().item()

            elif method == 'logit_gap':
                logits = likelihood_dict.get('logits')
                if isinstance(logits, tuple):
                    logits = logits[0]
                if logits is None or logits.dim() == 3:
                    logits = logits[0]
                if logits.dim() != 2:
                    print(f"⚠️ logit_gap: unexpected logits shape: {logits.shape}")
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
                question_emb = output.get("question_embedding")
                if question_emb is None:
                    print("⚠️ Missing question_embedding for AWARE.")
                    scores['aware'] = float("nan")
                else:
                    scores['aware'] = compute_aware_uncertainty(likelihood_dict, output, question_emb)

            else:
                print(f"⚠️ Unknown uncertainty method: {method}")
                scores[method] = float("nan")

        except Exception as e:
            print(f"⚠️ Failed computing {method}: {e}")
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
        input_text = output.get("input_text", "").strip()
        answer_text = output.get("generated_text", "").strip()

        if not input_text or not answer_text:
            print(f"⚠️ BERT-SAR skipped: empty input or output. input='{input_text}', answer='{answer_text}'")
            return float("nan")

        tokens = input_text.split()
        if len(tokens) == 0:
            print(f"⚠️ BERT-SAR skipped: no tokens in input_text='{input_text}'")
            return float("nan")

        token_embeddings = sbert.encode(tokens, convert_to_tensor=True, normalize_embeddings=True)
        answer_embedding = sbert.encode(answer_text, convert_to_tensor=True, normalize_embeddings=True)

        similarities = util.pytorch_cos_sim(token_embeddings, answer_embedding).squeeze()
        similarities = similarities[:len(log_likelihoods)]

        if similarities.sum().item() == 0 or similarities.numel() == 0:
            print(f"⚠️ BERT-SAR warning: similarity sum zero for input='{input_text}'")
            return float("nan")

        weights = similarities / similarities.sum().clamp(min=1e-6)
        score = -(log_likelihoods[:len(weights)] * weights).sum().item()

        return score
    except Exception as e:
        print(f"⚠️ BERT-SAR error: {e}")
        return float("nan")
