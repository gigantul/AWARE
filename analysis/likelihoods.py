import torch
import torch.nn.functional as F
from typing import Dict

def compute_likelihoods(output):
    # Handle too-short generations
    if len(output.get("scores", [])) < 2:
        print(f"⚠️ Skipping: not enough tokens to compute likelihoods. Generated: {output['generated_text']}")
        return {
            "token_log_likelihoods": torch.tensor([]),
            "entropy_per_token": torch.tensor([]),
            "log_probs": torch.tensor([])
        }

    # Convert to CPU for safe debugging
    logits = torch.stack(output["scores"]).float().cpu()
    num_logits = logits.shape[0]

    # Align labels with logits: keep only the last N tokens where N = # of logits
    # This trims off prompt tokens and avoids mismatch in likelihood computation
    labels = output["generated_ids"][-num_logits:].cpu() if num_logits <= len(output["generated_ids"]) else output[
        "generated_ids"].cpu()

    # Shift logits and labels to align next-token prediction
    shifted_logits = logits[:-1]
    shifted_labels = labels

    if shifted_logits.size(0) != shifted_labels.size(0):
        print(f"⚠️ Skipping: shifted logits/labels mismatch. logits={shifted_logits.size(0)}, labels={shifted_labels.size(0)}")
        return {
            "token_log_likelihoods": torch.tensor([]),
            "entropy_per_token": torch.tensor([]),
            "log_probs": torch.tensor([])
        }

    log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)

    vocab_size = log_probs.shape[-1]
    if shifted_labels.max() >= vocab_size or shifted_labels.min() < 0:
        print(f"⚠️ Skipping: label index out of bounds. max={shifted_labels.max()}, vocab={vocab_size}")
        return {
            "token_log_likelihoods": torch.tensor([]),
            "entropy_per_token": torch.tensor([]),
            "log_probs": log_probs
        }

    try:
        token_log_likelihoods = log_probs[range(len(shifted_labels)), shifted_labels]
    except IndexError as e:
        print(f"⚠️ Skipping: IndexError in token log-likelihoods lookup. {str(e)}")
        return {
            "token_log_likelihoods": torch.tensor([]),
            "entropy_per_token": torch.tensor([]),
            "log_probs": log_probs
        }

    entropy_per_token = -torch.sum(torch.exp(log_probs) * log_probs, dim=-1)

    return {
        "token_log_likelihoods": token_log_likelihoods,
        "entropy_per_token": entropy_per_token,
        "log_probs": log_probs
    }
