from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from mini_llm.inference.engine.llm_engine import LLMEngine
from mini_llm.inference.model.gpt import GPT2
from mini_llm.inference.model.weight_loader import load_hf_gpt2_weights


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "mini_llm"
    / "inference"
    / "model"
    / "tiny-gpt2"
)

def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_custom_model(hf_model):
    config = hf_model.config

    custom_model = GPT2(
        vocab_size=config.vocab_size,
        max_seq_len=config.n_positions,
        n_heads=config.n_head,
        dim=config.n_embd,
        hidden_dim=config.n_inner or 4 * config.n_embd,
        n_layers=config.n_layer,
    )

    load_hf_gpt2_weights(
        custom_model,
        hf_model,
    )

    return custom_model


def main():
    device = select_device()
    print(f"device: {device}")
    print(f"loading: {MODEL_PATH}")

    hf_model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_custom_model(hf_model)
    model.to(device)
    model.eval()

    engine = LLMEngine(
        model=model,
        max_seq_len=32,
        batch_size=2,
        end_token_id=tokenizer.eos_token_id,
    )

    engine.model_runner.pad_token_id = tokenizer.pad_token_id

    prompts = {
        "hello": "Hello",
        "story": "Once upon a time",
        "meaning": "The meaning of life is",
    }

    for request_id, prompt in prompts.items():
        prompt_token_ids = tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )
        engine.add_request(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
        )

    engine.generate()

    print("\n=== results ===")
    for seq in engine.scheduler.cq:
        text = tokenizer.decode(
            seq.token_ids,
            skip_special_tokens=True,
        )
        print(f"[{seq.request_id}] {text}")


if __name__ == "__main__":
    main()
