from typing import Tuple, List
from vllm import LLM
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from numpy.lib.function_base import gradient
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from torch.nn.utils.rnn import pad_sequence
from unittest.mock import patch
import torch

def tokenize_prompt_and_output(prompt_strs: List[str],
                               output_strs: List[str],
                               tokenizer: PreTrainedTokenizer) ->dict[str, torch.Tensor]:
    seqs = []
    masks = []
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        # common fallback in GPT-style tokenizers
        pad_id = tokenizer.eos_token_id

    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)
        output_ids = tokenizer(output, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)

        full_ids = torch.cat([prompt_ids, output_ids], dim=0)  # (Lp+Lo,)
        seqs.append(full_ids)

        # token-level mask: 0 for prompt tokens, 1 for output tokens
        token_mask = torch.cat(
            [torch.zeros_like(prompt_ids), torch.ones_like(output_ids)],
            dim=0
        )
        masks.append(token_mask)

    padded_ids = pad_sequence(seqs, batch_first=True, padding_value=pad_id)  # (B, Tmax)
    padded_mask = pad_sequence(masks, batch_first=True, padding_value=0)  # (B, Tmax)

    input_ids = padded_ids[:, :-1]  # (B, Tmax-1)
    labels = padded_ids[:, 1:]  # (B, Tmax-1)
    response_mask = padded_mask[:, 1:]  # align mask with labels

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def compute_entropy(logits: torch.Tensor)->torch.Tensor:
    """
    logits: tensor of shape (batch_size, seq_len, vocab_size)
    return : tensor of shape (batch_size, seq_len) containing entropy for next token prediction
    """
    Z = torch.logsumexp(logits, dim=-1, keepdim=True)
    log_p = logits - Z
    return -torch.sum(torch.exp(log_p) * log_p, dim=-1)


def get_response_log_probs(
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        return_token_entropy: bool = False,
    ) -> dict[str, torch.Tensor]:
    """
    Compute log probabilities of the response tokens given the prompt tokens.
    Args:
        model: The model to use for inference.
        input_ids: The input IDs of the prompt tokens.
        labels: The labels for computing the loss.
        return_token_entropy: Whether to return the entropy of the next token prediction.
    Returns:
        dict[str, torch.Tensor]
        "log_probs" of shape (batch_size, seq_len)
        "token_entropy" of shape (batch_size, seq_len)
    """
    logits = model(input_ids).logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1) # (batch_size, seq_len, vocab)
    response_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        return {"log_probs": response_log_probs, "token_entropy": token_entropy}
    else:
        return {"log_probs": response_log_probs}


def masked_normalize(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        normalize_constant: float,
        dim: int | None = None,
    ) -> torch.Tensor:
    """
    Sum over a dimension and normalize by a constant, considering only the elements with mask value 1.
    Args:
        tensor: The tensor to sum and normalize.
        mask: The mask. We only consider elements with mask value 1.
        normalize_constant: The constant to divide by for normalization.
        dim: The dimension to sum along before normalization. If None, sum over all dimensions.
    Returns:
        torch.Tensor the normalized sum
    """
    masked_tensor = torch.where(mask, tensor, 0)
    tensor_sum = masked_tensor.sum(dim=dim)
    return tensor_sum / normalize_constant


def sft_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        normalize_constant: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.
    Args:
        policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
        SFT policy being trained.
        response_mask (batch_size, sequence_length), 1 for response tokens, 0 for
        prompt/padding.
        gradient_accumulation_steps Number of microbatches per optimizer step.
        normalize_constant The constant by which to divide the sum. It is fine to leave this as 1.0.
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
            this so we can log it.
            metadata Dict with metadata from the underlying loss call, and any other statistics you
            might want to log.
    """
    loss = - masked_normalize(policy_log_probs, response_mask, normalize_constant, dim=-1) / gradient_accumulation_steps
    loss = loss.mean()
    loss.backward()
    return loss, {"loss": loss.item(),
                  "gradient_accumulation": gradient_accumulation_steps}

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
    return_value=None
    )
    with world_size_patch, profiling_patch:
    return LLM(
    model=model_id,
    device=device,
    dtype=torch.bfloat16,
    enable_prefix_caching=True,
    gpu_memory_utilization=gpu_memory_utilization,
    )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())
